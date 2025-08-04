from functools import partial
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=40, help="number of grid points across one axis per subdomain")
parser.add_argument("--M", type=partial(int, base=0), default=1024, help="number of neurons per subdomain")
parser.add_argument(
  "--problem", "-p", type=str, default='sin',
  choices=[
    'sin', 'sin2piexp', 'grf_32_0b'
  ],
  help="select problem parameters"
)
parser.add_argument("--wct", type=int, default=0, help="get average wall clock time of specified runs")
parser.add_argument("--error_n", type=int, default=None, help="number of points to evaluate error")
parser.add_argument("--chunk_size", type=int, default=None, help="2 ** chunk size for vmap")
parser.add_argument("--tol", type=float, default=1e-9, help="rel tol for cg")
parser.add_argument("--atol", type=float, default=0, help="abs tol for cg")
parser.add_argument("--maxiter", type=int, default=None, help="maxiter for cg")
parser.add_argument("--solver", type=str, default="ykdd", choices=["ykdd", "neum", "elm", "table_theta", "table_k2a2"], help="select solver to use")
parser.add_argument("--save_img_paper", action='store_true', help="save image of solution for paper")
parser.add_argument("--plot_n", type=int, default=None, help="number of points to plot")
parser.add_argument("--lin_solver", type=str, default="cg", choices=["cg", "cg_cond"], help="select linear solver to use")
parser.add_argument("--theta", type=float, default=1, help="theta for neumann-neumann")
parser.add_argument("--l", type=float, default=None, help="length for initialization")
parser.add_argument("--foam", type=float, default=None, help="foam for initialization")
args = parser.parse_args()

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import torch
import numpy as np
from typing import Callable, Union, NamedTuple
from abc import ABC, abstractmethod
from torch.func import vmap, vjp, jvp, jacrev, hessian, jacfwd

Tensor = torch.Tensor
dtype = torch.float64
ones = partial(torch.ones, dtype=dtype)
zeros = partial(torch.zeros, dtype=dtype)
empty = partial(torch.empty, dtype=dtype)
linspace = partial(torch.linspace, dtype=dtype)
eye = partial(torch.eye, dtype=dtype)

from numpy import pi

N = int(size ** .5)
idx, jdx = rank // N, rank % N

import time
from pathlib import Path

import runtime_constants
chunk_size = 2 ** args.chunk_size if args.chunk_size is not None else None
runtime_constants.chunk_size = chunk_size

import base_class

from linear_solvers import cg_distributed
from linear_solvers import get_linear_solver

act = torch.tanh
_d_act = jacrev(act)
_dd_act = jacrev(_d_act)
d_act = vmap(_d_act)
dd_act = vmap(_dd_act)
class PoissonYKDD(base_class.YKDD):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.act = act

  def Lnet_x_single(self, x: Tensor, theta) -> Tensor:
    return -dd_act(x @ theta['w'] + theta['b']) * (theta['w'] * theta['w']).sum(dim=0)
  
  def fluxnet_x_single(self, x: Tensor, theta, flux_info: Tensor) -> Tensor:
    return d_act(x @ theta['w'] + theta['b']) * torch.einsum('dM, d->M', theta['w'], flux_info)

  def ykdd_table_k2a2(self, grid: base_class.Grid, theta, *, driver='gelsd', tol=1e-9, atol=0, lin_solver='cg', Theta=1, maxiter=None):
    eye = partial(torch.eye, dtype=dtype)
    
    K, (K_prim_size, K_dual_size), (prim_to_dof, dual_to_dof, glob_prim_to_dof) = self.get_K(grid, theta)
    A_prim, A_idcs_prim, A_dual, A_idcs_dual = self.get_A(grid, theta)
    B_prim = zeros(K.shape[0], K_prim_size)
    B_prim[K.shape[0]-K_prim_size-K_dual_size:K.shape[0]-K_dual_size] = -eye(K_prim_size)
    B_dual = zeros(K.shape[0], K_dual_size)
    B_dual[K.shape[0]-K_dual_size:] = -eye(K_dual_size)
    f = self.get_rhs(grid)
    mult_prim, mult_dual = self.get_multiplicity(grid)
  
    U, S, Vh = torch.linalg.svd(K, full_matrices=False)
    rcond = torch.finfo(K.dtype).eps * max(*K.shape)
    mask = S >= torch.tensor(rcond, dtype=K.dtype) * S[0]
    safe_idx = mask.sum()
    U, S, Vh = U[:, :safe_idx], S[:safe_idx], Vh[:safe_idx]
    S_inv = 1 / S

    K_tilde = torch.cat(
      [
        K[:K.shape[0]-K_dual_size],
        A_dual
      ]
    )
    U_tilde, S_tilde, Vh_tilde = torch.linalg.svd(K_tilde, full_matrices=False)
    rcond = torch.finfo(K_tilde.dtype).eps * max(*K_tilde.shape)
    mask = S_tilde >= torch.tensor(rcond, dtype=K_tilde.dtype) * S_tilde[0]
    safe_idx = mask.sum()
    U_tilde, S_tilde, Vh_tilde = U_tilde[:, :safe_idx], S_tilde[:safe_idx], Vh_tilde[:safe_idx]
    S_inv_tilde = 1 / S_tilde

    K_o, _, _ = self.get_K_o(grid, theta)
    A_prim_o, _, A_dual_o, _ = self.get_A_o(grid, theta)

    U_o, S_o, Vh_o = torch.linalg.svd(K_o, full_matrices=False)
    rcond = torch.finfo(K_o.dtype).eps * max(*K_o.shape)
    mask = S_o >= torch.tensor(rcond, dtype=K_o.dtype) * S_o[0]
    safe_idx = mask.sum()
    U_o, S_o, Vh_o = U_o[:, :safe_idx], S_o[:safe_idx], Vh_o[:safe_idx]
    S_inv_o = 1 / S_o

    K_tilde_o = torch.cat(
      [
        K_o[:K_o.shape[0]-K_dual_size],
        A_dual_o
      ]
    )
    U_tilde_o, S_tilde_o, Vh_tilde_o = torch.linalg.svd(K_tilde_o, full_matrices=False)
    rcond = torch.finfo(K_tilde_o.dtype).eps * max(*K_tilde_o.shape)
    mask = S_tilde_o >= torch.tensor(rcond, dtype=K_tilde_o.dtype) * S_tilde_o[0]
    safe_idx = mask.sum()
    U_tilde_o, S_tilde_o, Vh_tilde_o = U_tilde_o[:, :safe_idx], S_tilde_o[:safe_idx], Vh_tilde_o[:safe_idx]
    S_inv_tilde_o = 1 / S_tilde_o

    def k2a2_sub(K, U, S_inv, Vh, U_tilde, S_inv_tilde, Vh_tilde, A_prim):
      A_tilde = K[K.shape[0]-K_dual_size:]
      B_tilde = zeros(K_tilde.shape[0], A_dual.shape[0])
      B_tilde[K.shape[0]-K_dual_size:] = -eye(A_dual.shape[0])
    
      AKpB_tilde = A_tilde @ (Vh_tilde.T @ (torch.einsum('i, i...->i...', S_inv_tilde, U_tilde.T @ B_tilde)))

      BT_idcs_prim = [prim_to_dof(interface.idcs) for interface in grid.prim_intfc]
      BT_idcs_dual = [dual_to_dof(interface.idcs) for interface in grid.dual_intfc]

      UTB_prim, UTB_dual = U.T @ B_prim, U.T @ B_dual

      prim_size = prim_to_dof(torch.arange(grid.prim_size)).shape[0]
      prim_idcs = glob_prim_to_dof(grid.prim_to_glob)

      KpB_prim = Vh.T @ (torch.einsum('i, i...->i...', S_inv, UTB_prim))
      KpB_dual = Vh.T @ (torch.einsum('i, i...->i...', S_inv, UTB_dual))
      AKpB_prim_local = zeros(A_prim.shape[0], prim_size)
      AKpB_prim_local[:, prim_idcs] = A_prim @ KpB_prim
      AKpB_prim = self.A_comms(grid, AKpB_prim_local, A_idcs_prim, prim=True)

      prim_prob_local = eye(B_prim.shape[1]) # BTB
      prim_prob_local -= UTB_prim.T @ UTB_prim # BTKKpB

      prim_mat_idcs = torch.cartesian_prod(prim_idcs, prim_idcs)

      prim_prob_send = AKpB_prim_local.T @ AKpB_prim
      prim_prob_send[prim_mat_idcs[:, 0], prim_mat_idcs[:, 1]] += prim_prob_local.reshape(-1)
      prim_prob = empty(prim_size, prim_size) if rank == 0 else None
      comm.Reduce(prim_prob_send, prim_prob, op=MPI.SUM, root=0)
      if rank == 0:
        prim_prob_cho, cho_info = torch.linalg.cholesky_ex(prim_prob)
      def prim_prob_solve(b: Tensor):
        b_send = zeros(prim_size, *b.shape[1:])
        b_send[prim_idcs] = b
        b_recv = empty(prim_size, *b.shape[1:])
        comm.Reduce(b_send, b_recv, op=MPI.SUM, root=0)
        if rank == 0:
          rax = torch.cholesky_solve(b_recv.reshape(prim_size, -1), prim_prob_cho).reshape(b_recv.shape)
        else:
          rax = empty(b_recv.shape)
        comm.Bcast(rax, root=0)
        return rax

      D_pp = -AKpB_prim_local[:, prim_idcs]
      D_pp_glob = -AKpB_prim
      D_pd = -A_prim @ KpB_dual
      D_dp = -A_dual @ KpB_prim
      D_dd = -A_dual @ KpB_dual
      BTUUTB_pd = UTB_prim.T @ UTB_dual
      BTUUTB_dp = UTB_dual.T @ UTB_prim
      BTUUTB_dd = UTB_dual.T @ UTB_dual

      def E(u, D_pdu):
        mu_prim = prim_prob_solve(-BTUUTB_pd @ u + D_pp.T @ D_pdu)
        mu_glob, mu_prim = mu_prim, mu_prim[prim_idcs]
        n = self.A_comms(grid, D_dd @ u - D_dp @ mu_prim, A_idcs_dual, prim=False)
        return n, mu_glob, mu_prim
      
      def ET(n):
        D_dpTn = D_dp.T @ n
        nu_prim = prim_prob_solve(D_dpTn)
        nu_glob, nu_prim = nu_prim, nu_prim[prim_idcs]
        rax = D_dd.T @ n + BTUUTB_dp @ nu_prim - (D_pd.T @ (D_pp_glob @ nu_glob))
        return rax
      
      def M(n):
        u = AKpB_tilde @ n
        return self.BT_comms(grid, u, BT_idcs_dual, prim=False)
      
      def MT(u):
        n = AKpB_tilde.T @ u
        return self.A_comms(grid, n, A_idcs_dual, prim=False)
      
      def ykdd_table_theta_sub(Theta):
        def apply_mat(u: Tensor) -> Tensor:
          D_pdu = self.A_comms(grid, D_pd @ u, A_idcs_prim, prim=True)
          n, mu_glob, mu_prim = E(u, D_pdu)
          rax2 = ET(Theta * MT(M(n)) + (1 - Theta) * n)
          rax1 = u + D_pd.T @ D_pdu
          rax3 = BTUUTB_dd @ u - BTUUTB_dp @ mu_prim + (D_pd.T @ (D_pp_glob @ mu_glob))

          rax = self.BT_comms(grid, rax1 + rax2 - rax3, BT_idcs_dual, prim=False)
          return rax
        
        UTf = U.T @ f
        Kpf = Vh.T @ (torch.einsum('i, i...->i...', S_inv, UTf))
        d_p = -A_prim @ Kpf
        d_p = self.A_comms(grid, d_p, A_idcs_prim, prim=True)

        rhs_mu_prim = prim_prob_solve(-UTB_prim.T @ UTf + D_pp.T @ d_p)
        rhs_mu_glob, rhs_mu_prim = rhs_mu_prim, rhs_mu_prim[prim_idcs]
        rhs_n = self.A_comms(grid, -A_dual @ Kpf - D_dp @ rhs_mu_prim, A_idcs_dual, prim=False)
        rhs2 = ET(Theta * MT(M(rhs_n)) + (1 - Theta) * rhs_n)
        rhs1 = B_dual.T @ f + D_pd.T @ d_p
        rhs3 = UTB_dual.T @ UTf - BTUUTB_dp @ rhs_mu_prim + (D_pd.T @ (D_pp_glob @ rhs_mu_glob))

        rhs = self.BT_comms(grid, rhs1 + rhs2 - rhs3, BT_idcs_dual, prim=False)

        if type(lin_solver) is str:
          if lin_solver == 'cg':  
            u_d, info = cg_distributed(mult_dual, grid.dual_size, apply_mat, rhs, tol=tol, atol=atol, maxiter=maxiter)
          else:
            raise ValueError('lin_solver must be one of "cg", "gmres", "lanczos"')

        else:
          u_d, info = lin_solver(mult_dual, grid.dual_size, apply_mat, rhs)

        mu_prim = prim_prob_solve(-BTUUTB_pd @ u_d + D_pp.T @ self.A_comms(grid, D_pd @ u_d, A_idcs_prim, prim=True))
        mu_glob, mu_prim = mu_prim, mu_prim[prim_idcs]
        u_p = rhs_mu_prim - mu_prim
        c = (Vh.T @ (torch.einsum('i, i...->i...', S_inv, UTf - UTB_prim @ u_p - UTB_dual @ u_d)))

        return c, info
      
      for Theta in [0., 1.]:
        c, info = ykdd_table_theta_sub(Theta)
        if rank == 0:
          print(f'Theta: {Theta} | iters: {info[0]: 5d} | residual: {info[1]:.4e}')
        self.compute_errors(grid if args.error_n is None else ykdd.get_grid(n=args.error_n), c, theta, target_exact, target_exact_grad)
      
      return c, info
    
    k2a2_sub(K_o, U_o, S_inv_o, Vh_o, U_tilde_o, S_inv_tilde_o, Vh_tilde_o, A_prim_o)
    k2a2_sub(K_o, U_o, S_inv_o, Vh_o, U_tilde_o, S_inv_tilde_o, Vh_tilde_o, A_prim)
    k2a2_sub(K, U, S_inv, Vh, U_tilde, S_inv_tilde, Vh_tilde, A_prim_o)
    c, info = k2a2_sub(K, U, S_inv, Vh, U_tilde, S_inv_tilde, Vh_tilde, A_prim)

    return c, info
  
def get_problem(problem: str):
  def awb_from_path(path: str):
    dir = Path(path)
    a = torch.from_numpy(np.load(dir / 'a.npy'))
    w = torch.from_numpy(np.load(dir / 'w.npy'))
    b = torch.from_numpy(np.load(dir / 'b.npy'))
    return a, w, b
  
  def grf_coeffs(a, w, b):
    f_a, g_a, c_a = a[0], a[1], a[2]
    f_w, g_w, c_w = w[0], w[1], w[2]
    f_b, g_b, c_b = b[0], b[1], b[2]
    coeff = lambda x: torch.tanh(torch.sin(x @ c_w + c_b) @ c_a) + 1.1
    target_rhs = lambda x: torch.sin(x @ f_w + f_b) @ f_a
    target_bou = lambda x: torch.sin(x @ g_w + g_b) @ g_a
    return coeff, target_rhs, target_bou
  
  def get_fem_sols(u_fem, u_fem_grad, grad_idx=0):
    u_fem, u_fem_grad = torch.from_numpy(u_fem), torch.from_numpy(u_fem_grad)
    n = u_fem.shape[0] - 1
    target_exact = lambda x: u_fem[(n * x[..., 0]).int(), (n * x[..., 1]).int()]
    def target_exact_grad(x):
      shape = x.shape
      idcs = (n * x).int().reshape(-1, 2)
      grad_x = u_fem_grad[:, idcs[:, 0], idcs[:, 1]]
      grad_x = grad_x.transpose(0, 1).reshape(*shape[:-1], 2)
      return grad_x
    def target_exact_grad2(x):
      shape = x.shape
      idcs = (n * x).int().reshape(-1, 2)
      grad_x = u_fem_grad[idcs[:, 0], idcs[:, 1], :]
      grad_x = grad_x.reshape(*shape[:-1], 2)
      return grad_x
    return target_exact, (target_exact_grad if grad_idx == 0 else target_exact_grad2)
  
  def sin_prob(a: float):
    target_rhs = lambda x: (pi * a)**2 * 2 * (pi * a * x).sin().prod(dim=-1)
    target_bou = lambda x: (pi * a * x).sin().prod(dim=-1)
    target_exact = lambda x: (pi * a * x).sin().prod(dim=-1)
    target_exact_grad = base_class.act_last(jacrev(target_exact), 0, chunk_size=chunk_size)
    return target_rhs, target_bou, target_exact, target_exact_grad
  
  u_fem_slice, n_fem = None, None
  fenicsx_path = Path('fem_sols')

  if problem == 'sin':
    target_rhs, target_bou, target_exact, target_exact_grad = sin_prob(1)

  elif problem == 'sin2piexp':
    target_rhs = lambda x: ((2 * pi)**2 - 1) * (2 * pi * x[...,0]).sin() * (x[...,1]).exp()
    target_bou = lambda x: (2 * pi * x[...,0]).sin() * (x[...,1]).exp()
    target_exact = lambda x: (2 * pi * x[...,0]).sin() * (x[...,1]).exp()
    target_exact_grad = base_class.act_last(jacrev(target_exact), 0, chunk_size=chunk_size)

  elif problem[:3] == 'sin':
    target_rhs, target_bou, target_exact, target_exact_grad = sin_prob(float(problem[3:-2]))

  elif problem == 'grf_32_0b':
    a, w, b = awb_from_path('grf_coeffs/poisson/a32_M256_seed43')
    _, target_rhs, _ = grf_coeffs(a, w, b)
    target_bou = lambda x: zeros(x.shape[:-1])
    u_fem = np.load(fenicsx_path / 'u0_plot_grf_32_0b_32_16_1025.npy').reshape(1025, 1025)
    u_fem_grad = np.load(fenicsx_path / 'u0_plot_grf_32_0b_32_16_1025_grad.npy').reshape(1025, 1025, 2)
    target_exact, target_exact_grad = get_fem_sols(u_fem, u_fem_grad, grad_idx=1)

  return target_rhs, target_bou, target_exact, target_exact_grad
  
if __name__ == '__main__':
  t1 = time.time()
  target_rhs, target_bou, target_exact, target_exact_grad = get_problem(args.problem)
  
  ykdd = PoissonYKDD(
    target_rhs = target_rhs,
    target_bou = target_bou
  )

  n = args.n
  M = args.M
  grid = ykdd.get_grid(n=n)
  torch.manual_seed(rank+3)

  l = (M * size / 64) ** .5  if args.l is None else args.l
  w = (torch.rand(2, M, dtype=dtype) * 2 - 1) * l
  foam = 1/N if args.foam is None else args.foam
  b = -torch.einsum('Md, dM -> M', torch.rand(M, 2, dtype=dtype) * (1/N + foam) + torch.tensor([idx / N - foam / 2, jdx / N - foam / 2], dtype=dtype), w)

  if rank == 0:
    print(f'l {l:.2f} | foam {foam:.4f}')

  theta = {
    'w': w,
    'b': b
  }

  solver = {
    "ykdd": ykdd.ykdd,
    "neum": ykdd.ykdd_neum,
    "elm": ykdd.elm,
    "table_theta":ykdd.ykdd_table_theta,
    "table_k2a2": ykdd.ykdd_table_k2a2,
  }[args.solver]
  lin_param_dict = {
    "tol": args.tol,
    "atol": args.atol,
    "maxiter": args.maxiter,
  }
  lin_solver = get_linear_solver(args.lin_solver, **lin_param_dict)
  param_dict = {
    'lin_solver': lin_solver,
  }
  if args.solver in {'neum', 'direct', 'table_theta', 'table_k2a2'}:
    param_dict['Theta'] = args.theta
  if args.solver in {'table_theta'}:
    param_dict['error_function'] = lambda c: ykdd.compute_errors(
      grid if args.error_n is None else ykdd.get_grid(n=args.error_n),
      c, theta, target_exact, target_exact_grad
    )
  c, info = solver(grid, theta, **param_dict)
  t2 = time.time()
  
  if rank == 0:
    print(
      f'{args.lin_solver} ',
      f'iters: {info[0]: 5d} | residual: {info[1]:.4e}' if 'cg' in args.lin_solver else f'iters: {info[0]: 5d} | restarts: {info[1]: 5d} | residual: {info[2]:.4e}',
      f' | time to first: {t2 - t1:.2f}s | interface size: {grid.dual_size: 6d}',
      sep=''
    )

  ykdd.compute_errors(grid if args.error_n is None else ykdd.get_grid(n=args.error_n), c, theta, target_exact, target_exact_grad)
  if args.wct > 0:
    comm.barrier()
    t = time.time()
    for _ in range(args.wct):
      grid = ykdd.get_grid(n=n)
      solver(grid, theta, **param_dict)
    t1 = time.time()
    if rank == 0:
      print(f'{args.wct} runs | average wct: {(t1 - t)/args.wct:.2f}s')

  if args.save_img_paper:
    plot_n = args.plot_n if args.plot_n is not None else n
    plot_x = linspace(0, 1, plot_n+1) / N
    plot_x, plot_y = plot_x + idx / N, plot_x + jdx / N
    if idx > 0:
      plot_x[0] += 1 / plot_n / 10 / N
    if idx < N-1:
      plot_x[-1] -= 1 / plot_n / 10 / N
    if jdx > 0:
      plot_y[0] += 1 / plot_n / 10 / N
    if jdx < N-1:
      plot_y[-1] -= 1 / plot_n / 10 / N 
    plot_grid = torch.cartesian_prod(plot_x, plot_y)

    y = ykdd.net_xa(plot_grid, c, theta)
    y1_gather = empty(size, plot_n+1, plot_n+1) if rank == 0 else None
    comm.Gather(y.reshape(1, plot_n+1, plot_n+1), y1_gather, root=0)

  if args.save_img_paper:
    plot_n = args.plot_n if args.plot_n is not None else n
    plot_x = linspace(0, 1, plot_n+1) / N
    plot_x, plot_y = plot_x + idx / N, plot_x + jdx / N
    if idx > 0:
      plot_x[0] += 1 / plot_n / 10 / N
    if idx < N-1:
      plot_x[-1] -= 1 / plot_n / 10 / N
    if jdx > 0:
      plot_y[0] += 1 / plot_n / 10 / N
    if jdx < N-1:
      plot_y[-1] -= 1 / plot_n / 10 / N 
    plot_grid = torch.cartesian_prod(plot_x, plot_y)

    y = ykdd.net_xa(plot_grid, c, theta)
    y1_gather = empty(size, plot_n+1, plot_n+1) if rank == 0 else None
    comm.Gather(y.reshape(1, plot_n+1, plot_n+1), y1_gather, root=0)

    if rank == 0:
      import matplotlib as mpl
      import matplotlib.pyplot as plt
      y1_all = empty((plot_n + 1) * N, (plot_n + 1) * N)
      for i in range(N):
        for j in range(N):
          y1_all[i*(plot_n+1):(i+1)*(plot_n+1), j*(plot_n+1):(j+1)*(plot_n+1)] = y1_gather[i*N + j]

      plot_x = linspace(0, 1, plot_n + 1).reshape(1, -1) / N + linspace(0, 1, N + 1)[:-1].reshape(-1, 1)
      plot_x[1:, 0] += 1 / plot_n / 10 / N
      plot_x[:-1, -1] -= 1 / plot_n / 10 / N

      plot_x = plot_x.flatten()
      plot_y = plot_x
      plot_grid = torch.cartesian_prod(plot_x, plot_y)
      plot_n_all = (plot_n + 1) * N

      uses_fem_ref = args.problem in {'grf_32_0b'}
      if uses_fem_ref:
        y_ref = np.load(f'./fem_sols_revision/u0_plot_{args.problem}_32_16_{N}_{plot_n}.npy')
        y_ref = torch.tensor(y_ref).reshape(plot_n_all, plot_n_all)
      else:
        y_ref = target_exact(plot_grid).reshape(plot_n_all, plot_n_all)
      y_error = (y_ref - y1_all).abs()
      
      mpl.rcParams['figure.dpi'] = 600
      # plt.rcParams['text.usetex'] = True
      save_format = 'png'

      color_palette = 'plasma'
      # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
      def truncate_colormap(cmap, minval=0.0, maxval=1.0, N=256, n=256, gamma=1.0):
          new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)),
            N=N, gamma=gamma
          )
          return new_cmap

      arr = np.linspace(0, 50, 100).reshape((10, 10))
      fig, ax = plt.subplots(ncols=2)

      cmap = plt.get_cmap(color_palette)
      new_cmap = truncate_colormap(cmap, 0., 1., N=1024, gamma=1.)
      log_norm = lambda y: mpl.colors.LogNorm(vmin=y.min(), vmax=y.max())

      savefig = lambda title: plt.savefig(f'figures/poisson_{args.problem}_{N}_{args.plot_n}_{title}.{save_format}', format=save_format, bbox_inches='tight')

      def plot_2d(y, title, filename, title2=None, cmap='viridis', norm=None):
        fig, ax = plt.subplots()
        cs = ax.pcolormesh(plot_x, plot_y, y.T, shading='nearest', cmap=cmap, norm=norm)
        cb = plt.colorbar(cs)
        if norm is None:
          cb.formatter.set_powerlimits((-1, 1))
          cb.update_ticks()
        plt.axis('square')
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(title if uses_fem_ref else (title2 if title2 is not None else title))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        savefig(filename)
        return cs

      plot_2d(y1_all, 'Predicted solution', '2d')
      plot_2d(y_error, 'Absolute error (log scale)', '2d_error', cmap=new_cmap, norm=log_norm(y_error))
      plot_2d(y_ref, 'Reference solution', '2d_exact', 'Exact solution')

      fig, ax = plt.subplots()
      plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
      ax.plot(plot_x, y1_all.diag())
      ax.set_xlabel('$x$')
      ax.set_title('Predicted solution')
      savefig('diag')

      fig, ax = plt.subplots()
      plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
      ax.plot(plot_x, y_error.diag())
      ax.set_xlabel('$x$')
      ax.set_title('Absolute error')
      savefig('diag_error')

      fig, ax = plt.subplots()
      plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
      ax.plot(plot_x, y1_all.diag(), label='Predicted solution')
      ax.plot(plot_x, y_ref.diag(), label='Reference solution' if uses_fem_ref else 'Exact solution', linestyle='--')
      ax.set_xlabel('$x$')
      ax.set_title('Predicted vs ' + ('Reference solution' if uses_fem_ref else 'Exact solution') + ' along $x=y$')
      ax.legend()
      savefig('diag_exact')

      plot_x_3d, plot_y_3d = torch.meshgrid(plot_x, plot_y, indexing='xy')
      def plot_3d(y, title, filename, *, title2=None, stride=None, cmap='viridis'):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.ticklabel_format(axis='z', style='sci', scilimits=(-1,1))
        if stride is not None:
          surf = ax.plot_surface(plot_x_3d, plot_y_3d, y.T, rstride=stride, cstride=stride, cmap=cmap, antialiased=False)
        else:
          surf = ax.plot_surface(plot_x_3d, plot_y_3d, y.T, cmap=cmap, antialiased=False)

        # https://stackoverflow.com/questions/68143699/how-to-rotate-the-offset-text-in-a-3d-plot
        ax.zaxis.get_offset_text().set_visible(False)
        exponent = int('{:.2e}'.format(y.max()).split('e')[1])
        if exponent != 0:
          ax.text(ax.get_xlim()[1]*1.1, ax.get_ylim()[1], ax.get_zlim()[1]*1.1,
            '$\\times\\mathdefault{10^{%d}}\\mathdefault{}$' % exponent)

        cax = fig.add_axes([0.88, 0.1, 0.03, 0.8])
        cb = fig.colorbar(surf, cax=cax)
        cb.formatter.set_powerlimits((-1, 1))
        cb.update_ticks()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(title if uses_fem_ref else (title2 if title2 is not None else title))
        savefig(filename)

      plot_3d(y1_all, 'Predicted solution', '3d', stride=1)
      plot_3d(y_error, 'Absolute error', '3d_error', stride=1, cmap=new_cmap)
      plot_3d(y_ref, 'Reference solution', '3d_exact', stride=1, title2='Exact solution')