import torch
import numpy as np
from typing import Callable, Union, NamedTuple
from abc import ABC, abstractmethod
from torch.func import vmap, vjp, jvp, jacrev, hessian, jacfwd
from functools import partial, wraps
from inspect import signature

Tensor = torch.Tensor
dtype = torch.float64
ones = partial(torch.ones, dtype=dtype)
zeros = partial(torch.zeros, dtype=dtype)
empty = partial(torch.empty, dtype=dtype)
linspace = partial(torch.linspace, dtype=dtype)

from numpy import pi

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
N = int(size ** .5)
idx, jdx = rank // N, rank % N

import argparse
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, help="number of grid points across one axis per subdomain")
parser.add_argument("--M", type=partial(int, base=0), help="number of neurons per subdomain")
parser.add_argument("--error_n", type=int, default=None, help="number of grid points across one axis per subdomain for compute_error")
parser.add_argument(
  "--problem", "-p", type=str, default='sin',
  choices=[
    'sin'
  ],
  help="select problem parameters"
)
parser.add_argument("--wct", type=int, default=0, help="get average wall clock time of specified runs")
parser.add_argument("--chunk_size", type=int, default=None, help="2 ** chunk size for vmap")
parser.add_argument("--tol", type=float, default=1e-9, help="rel tol for cg")
parser.add_argument("--atol", type=float, default=0, help="abs tol for cg")
parser.add_argument("--maxiter", type=int, default=None, help="maxiter for cg")
parser.add_argument("--lin_solver", type=str, default="cg", choices=["cg", "cg_cond"], help="select linear solver to use")
parser.add_argument("--solver", type=str, default="ykdd", choices=["ykdd", "neum", "table_theta"], help="select solver to use")
parser.add_argument("--theta", type=float, default=1, help="theta for neumann-neumann")
parser.add_argument("--save_img_paper", action='store_true', help="save image of solution for paper")
parser.add_argument("--plot_n", type=int, default=None, help="number of points to plot")
parser.add_argument("--l", type=float, default=None, help="length for initialization")
parser.add_argument("--foam", type=float, default=None, help="foam for initialization")
args = parser.parse_args()

import runtime_constants
chunk_size = 2 ** args.chunk_size if args.chunk_size is not None else None
runtime_constants.chunk_size = chunk_size

import base_class

from linear_solvers import get_linear_solver

act = torch.tanh
_d_act = jacrev(act)
_dd_act = jacrev(_d_act)
_ddd_act = jacrev(_dd_act)
_dddd_act = jacrev(_ddd_act)
d_act = vmap(_d_act)
dd_act = vmap(_dd_act)
ddd_act = vmap(_ddd_act)
dddd_act = vmap(_dddd_act)
class BihYKDD(base_class.YKDD):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.act = act
    self.target_bou_lap = kwargs['target_bou_lap']

  def Lap_f(self, x: Tensor, theta) -> Tensor:
    w, b = theta['w'], theta['b']
    return dd_act(x @ w + b) * (w * w).sum(dim=0)

  def Lnet_x_single(self, x: Tensor, theta) -> Tensor:
    w, b = theta['w'], theta['b']
    ww = w * w
    return dddd_act(x @ w + b) * ((ww * ww).sum(dim=0) + ww.prod(dim=0) * 2)
  
  def fluxnet_x_single(self, x: Tensor, theta, flux_info: Tensor) -> Tensor:
    w, b = theta['w'], theta['b']
    return d_act(x @ w + b) * torch.einsum('dM, d->M', w, flux_info)
  
  def fluxLapnet_x_single(self, x: Tensor, theta, flux_info: Tensor) -> Tensor:
    w, b = theta['w'], theta['b']
    return ddd_act(x @ w + b) * torch.einsum('M, dM, d->M', (w * w).sum(dim=0), w, flux_info)
  
  @partial(base_class.act_last, where_x=1)
  def Lapnet_x(self, x: Tensor, theta) -> Tensor:
    return self.Lap_f(x, theta)
  
  @partial(base_class.act_last, where_x=[1, 3])
  def fluxLapnet_x(self, x: Tensor, theta, flux_info: Tensor) -> Tensor:
    return self.fluxLapnet_x_single(x, theta, flux_info)

  def get_K(self, grid: base_class.Grid, theta) -> tuple[Tensor, int, Callable[[Tensor], Tensor]]:
    # corner change of variables
    N = int(size ** .5)
    i, j = rank // N, rank % N
    n = grid.n
    bou_to_dual = grid.bou_to_dual
    corner_change = []
    # up left; sends tag 0
    if i > 0 and j > 0:
      idcs_to_bou = torch.cat([torch.arange(3*n+1, 4*n), torch.arange(1, n)])
      change = torch.cat([linspace(0, -1, n+1)[1:-1], linspace(-1, 0, n+1)[1:-1]])
      corner_change.append((bou_to_dual[idcs_to_bou], change))
    # up right; sends tag 1
    if i > 0 and j < N-1:
      idcs_to_bou = torch.cat([torch.arange(1, n), torch.arange(n+1, 2*n)])
      change = torch.cat([linspace(0, -1, n+1)[1:-1], linspace(-1, 0, n+1)[1:-1]])
      corner_change.append((bou_to_dual[idcs_to_bou], change))
    # down right; sends tag 2
    if i < N-1 and j < N-1:
      idcs_to_bou = torch.cat([torch.arange(n+1, 2*n), torch.arange(2*n+1, 3*n)])
      change = torch.cat([linspace(0, -1, n+1)[1:-1], linspace(-1, 0, n+1)[1:-1]])
      corner_change.append((bou_to_dual[idcs_to_bou], change))
    # down left; sends tag 3
    if i < N-1 and j > 0:
      idcs_to_bou = torch.cat([torch.arange(2*n+1, 3*n), torch.arange(3*n+1, 4*n)])
      change = torch.cat([linspace(0, -1, n+1)[1:-1], linspace(-1, 0, n+1)[1:-1]])
      corner_change.append((bou_to_dual[idcs_to_bou], change))

    K_prim = self.net_x(grid.prim, theta)
    K_dual = self.net_x(grid.dual, theta)
    for i, change in enumerate(corner_change):
      idcs, change = change
      K_dual[idcs] += change.reshape(-1, 1) @ K_prim[i:i+1]

    def test_flux(x: Tensor) -> Tensor:
      rax = torch.tensor([-1, 0], dtype=dtype).expand(1, 2) * torch.eq(x[...,0], 0).unsqueeze(1)
      rax += torch.tensor([0, -1], dtype=dtype).expand(1, 2) * torch.eq(x[...,1], 0).unsqueeze(1)
      rax += torch.tensor([1, 0], dtype=dtype).expand(1, 2) * torch.eq(x[...,0], 1).unsqueeze(1)
      rax += torch.tensor([0, 1], dtype=dtype).expand(1, 2) * torch.eq(x[...,1], 1).unsqueeze(1)
      return rax
    
    KLap_prim = self.Lapnet_x(grid.prim, theta)
    KLap_dual = self.Lapnet_x(grid.dual, theta)
    for i, change in enumerate(corner_change):
      idcs, change = change
      KLap_dual[idcs] += change.reshape(-1, 1) @ KLap_prim[i:i+1]

    K = torch.cat(
      [
        self.Lnet_x(grid.int, theta),
        self.net_x(grid.bou, theta),
        self.Lapnet_x(grid.bou, theta),
        K_prim,
        KLap_prim,
        K_dual,
        KLap_dual
      ]
    )

    def prim_idcs_to_dof_idcs(idcs: Tensor) -> Tensor:
      return torch.cat(
        [
          idcs,
          idcs + grid.prim.shape[0]
        ]
      )
    
    def dual_idcs_to_dof_idcs(idcs: Tensor) -> Tensor:
      return torch.cat(
        [
          idcs,
          idcs + grid.dual.shape[0]
        ]
      )
    
    def glob_prim_idcs_to_glob_dof_idcs(idcs: Tensor) -> Tensor:
      return torch.cat(
        [
          2 * idcs,
          2 * idcs + 1
        ]
      )
    
    return K, (2 * grid.prim.shape[0], 2 * grid.dual.shape[0]), (prim_idcs_to_dof_idcs, dual_idcs_to_dof_idcs, glob_prim_idcs_to_glob_dof_idcs)
  
  def get_rhs(self, grid: base_class.Grid) -> Tensor:
    rhs = torch.cat(
      [
        self.target_rhs(grid.int),
        self.target_bou(grid.bou),
        self.target_bou_lap(grid.bou),
        zeros(grid.prim.shape[0]),
        zeros(grid.prim.shape[0]),
        zeros(grid.dual.shape[0]),
        zeros(grid.dual.shape[0])
      ]
    )

    return rhs

  def get_A(self, grid: base_class.Grid, theta) -> tuple[Tensor, list[int, int]]:
    N = int(size ** .5)
    i, j = rank // N, rank % N
    n = grid.n
    bou_to_dual = grid.bou_to_dual
    bou_to_prim = grid.bou_to_prim

    idx = 0
    idcs_prim = []
    empty_flux = empty(0, b.shape[0])
    A_prim = [empty_flux]
    if i > 0 and j > 0:
      A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append(torch.cat([
        self.fluxnet_x(grid.prim[bou_to_prim[0:1]], theta, torch.tensor([-1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[1:n]], theta, torch.tensor([-1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True),
        self.fluxLapnet_x(grid.prim[bou_to_prim[0:1]], theta, torch.tensor([-1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxLapnet_x(grid.dual[bou_to_dual[1:n]], theta, torch.tensor([-1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      ])/(n))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append(torch.cat([
        self.fluxnet_x(grid.prim[bou_to_prim[0:1]], theta, torch.tensor([0, -1], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[3*n+1:4*n]], theta, torch.tensor([0, -1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True),
        self.fluxLapnet_x(grid.prim[bou_to_prim[0:1]], theta, torch.tensor([0, -1], dtype=dtype).expand(1, 2)) +
        self.fluxLapnet_x(grid.dual[bou_to_dual[3*n+1:4*n]], theta, torch.tensor([0, -1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      ])/(n))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
    # up right; sends tag 1
    if i > 0 and j < N-1:
      A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append(torch.cat([
        self.fluxnet_x(grid.prim[bou_to_prim[n:n+1]], theta, torch.tensor([-1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[1:n]], theta, torch.tensor([-1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True),
        self.fluxLapnet_x(grid.prim[bou_to_prim[n:n+1]], theta, torch.tensor([-1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxLapnet_x(grid.dual[bou_to_dual[1:n]], theta, torch.tensor([-1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      ])/(n))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append(torch.cat([
        self.fluxnet_x(grid.prim[bou_to_prim[n:n+1]], theta, torch.tensor([0, 1], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[n+1:2*n]], theta, torch.tensor([0, 1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True),
        self.fluxLapnet_x(grid.prim[bou_to_prim[n:n+1]], theta, torch.tensor([0, 1], dtype=dtype).expand(1, 2)) +
        self.fluxLapnet_x(grid.dual[bou_to_dual[n+1:2*n]], theta, torch.tensor([0, 1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      ])/(n))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
    # down right; sends tag 2
    if i < N-1 and j < N-1:
      A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append(torch.cat([
        self.fluxnet_x(grid.prim[bou_to_prim[2*n:2*n+1]], theta, torch.tensor([1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[2*n+1:3*n]], theta, torch.tensor([1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True),
        self.fluxLapnet_x(grid.prim[bou_to_prim[2*n:2*n+1]], theta, torch.tensor([1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxLapnet_x(grid.dual[bou_to_dual[2*n+1:3*n]], theta, torch.tensor([1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      ])/(n))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append(torch.cat([
        self.fluxnet_x(grid.prim[bou_to_prim[2*n:2*n+1]], theta, torch.tensor([0, 1], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[n+1:2*n]], theta, torch.tensor([0, 1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True),
        self.fluxLapnet_x(grid.prim[bou_to_prim[2*n:2*n+1]], theta, torch.tensor([0, 1], dtype=dtype).expand(1, 2)) +
        self.fluxLapnet_x(grid.dual[bou_to_dual[n+1:2*n]], theta, torch.tensor([0, 1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      ])/(n))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
    # down left; sends tag 3
    if i < N-1 and j > 0:
      A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append(torch.cat([
        self.fluxnet_x(grid.prim[bou_to_prim[3*n:3*n+1]], theta, torch.tensor([1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[2*n+1:3*n]], theta, torch.tensor([1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True),
        self.fluxLapnet_x(grid.prim[bou_to_prim[3*n:3*n+1]], theta, torch.tensor([1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxLapnet_x(grid.dual[bou_to_dual[2*n+1:3*n]], theta, torch.tensor([1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      ])/(n))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append(torch.cat([
        self.fluxnet_x(grid.prim[bou_to_prim[3*n:3*n+1]], theta, torch.tensor([0, -1], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[3*n+1:4*n]], theta, torch.tensor([0, -1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True),
        self.fluxLapnet_x(grid.prim[bou_to_prim[3*n:3*n+1]], theta, torch.tensor([0, -1], dtype=dtype).expand(1, 2)) +
        self.fluxLapnet_x(grid.dual[bou_to_dual[3*n+1:4*n]], theta, torch.tensor([0, -1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      ])/(n))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]

    idx = 0
    idcs_dual = []
    A_dual = [empty(0, b.shape[0])]
    for interface in grid.dual_intfc:
      if interface.flux_info is not None:
        A_dual.append(torch.cat(
          [
            self.fluxnet_x(grid.dual[interface.idcs], theta, interface.flux_info),
            self.fluxLapnet_x(grid.dual[interface.idcs], theta, interface.flux_info)
          ]
        ))
      idcs_dual.append((idx, idx + A_dual[-1].shape[0]))
      idx = idcs_dual[-1][1]


    A_prim = torch.cat(A_prim)
    A_dual = torch.cat(A_dual)

    return A_prim, idcs_prim, A_dual, idcs_dual
  
  def get_multiplicity(self, grid: base_class.Grid) -> Tensor:
    return torch.cat([grid.prim_multiplicity, grid.prim_multiplicity]), torch.cat([grid.dual_multiplicity, grid.dual_multiplicity])
  
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
  
  def get_fem_sols(u_fem, u_fem_grad):
    u_fem, u_fem_grad = torch.from_numpy(u_fem), torch.from_numpy(u_fem_grad)
    n = u_fem.shape[0] - 1
    target_exact = lambda x: u_fem[(n * x[..., 0]).int(), (n * x[..., 1]).int()]
    def target_exact_grad(x):
      shape = x.shape
      idcs = (n * x).int().reshape(-1, 2)
      grad_x = u_fem_grad[:, idcs[:, 0], idcs[:, 1]]
      grad_x = grad_x.transpose(0, 1).reshape(*shape[:-1], 2)
      return grad_x
    return target_exact, target_exact_grad
  
  def sin_prob(a: float):
    D = 1 / (pi**4 * (2 * a**2)**2)
    target_rhs = lambda x: (pi * a * x).sin().prod(dim=-1) / D
    target_bou = lambda x: (pi * a * x).sin().prod(dim=-1)
    target_exact = lambda x: (pi * a * x).sin().prod(dim=-1)
    target_exact_grad = base_class.act_last(jacrev(target_exact), 0, chunk_size=chunk_size)
    return target_rhs, target_bou, target_exact, target_exact_grad
  
  target_bou_lap = lambda x: zeros(x.shape[0])

  if problem == 'sin':
    target_rhs, target_bou, target_exact, target_exact_grad = sin_prob(1)

  elif problem[:3] == 'sin':
    target_rhs, target_bou, target_exact, target_exact_grad = sin_prob(float(problem[3:-2]))

  else:
    raise NotImplementedError

  return target_rhs, target_bou, target_bou_lap, target_exact, target_exact_grad
  
if __name__ == '__main__':
  t1 = time.time()
  target_rhs, target_bou, target_bou_lap, target_exact, target_exact_grad = get_problem(args.problem)
  
  ykdd = BihYKDD(
    target_rhs = target_rhs,
    target_bou = target_bou,
    target_bou_lap = target_bou_lap,
  )

  n = args.n
  M = args.M
  grid = ykdd.get_grid(n=n)
  torch.manual_seed(rank+3)

  l = (M * size / 256) ** .5  if args.l is None else args.l
  w = (torch.rand(2, M, dtype=dtype) * 2 - 1) * l
  foam = 1/N if args.foam is None else args.foam
  b = -torch.einsum('Md, dM -> M', torch.rand(M, 2, dtype=dtype) * (1/N + foam) + torch.tensor([idx / N - foam / 2, jdx / N - foam / 2], dtype=dtype), w)

  if rank == 0:
    print(f'l {l:.2f} | foam {foam:.4f}')

  theta = {'w': w, 'b': b}

  solver = {
    "ykdd": ykdd.ykdd,
    "neum": ykdd.ykdd_neum,
    "table_theta":ykdd.ykdd_table_theta
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
  if args.solver in {'neum', 'direct', 'table_theta'}:
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

  error_grid = ykdd.get_grid(n=args.error_n) if args.error_n is not None else grid
  ykdd.compute_errors(error_grid, c, theta, target_exact, target_exact_grad)
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

      uses_fem_ref = args.problem in {}
      if uses_fem_ref:
        y_ref = np.load(f'./fem_sols_revision/u0_plot_var_{args.problem}_32_16_{plot_n}.npy')
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
      new_cmap = truncate_colormap(cmap, 0., 1., N=1024, gamma=.6)
      log_norm = lambda y: mpl.colors.LogNorm(vmin=y.min(), vmax=y.max())

      savefig = lambda title: plt.savefig(f'figures/bih_{args.problem}_{N}_{args.plot_n}_{title}.{save_format}', format=save_format, bbox_inches='tight')

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