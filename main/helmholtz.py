from functools import partial
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, help="number of grid points across one axis per subdomain")
parser.add_argument("--M", type=partial(int, base=0), help="number of neurons per subdomain")
parser.add_argument(
  "--problem", "-p", type=str, default='sin1pi4pi',
  choices=[
    'sin1pi4pi',
  ],
  help="select problem parameters"
)
parser.add_argument("--wct", type=int, default=0, help="get average wall clock time of specified runs")
parser.add_argument("--error_n", type=int, default=None, help="number of points to evaluate error")
parser.add_argument("--chunk_size", type=int, default=None, help="2 ** chunk size for vmap")
parser.add_argument("--tol", type=float, default=1e-9, help="rel tol for cg")
parser.add_argument("--atol", type=float, default=0, help="abs tol for cg")
parser.add_argument("--maxiter", type=int, default=None, help="maxiter for cg")
parser.add_argument("--solver", type=str, default="ykdd", choices=["ykdd", "neum", "table_theta"], help="select solver to use")
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

from linear_solvers import get_linear_solver

act = torch.tanh
_d_act = jacrev(act)
_dd_act = jacrev(_d_act)
d_act = vmap(_d_act)
dd_act = vmap(_dd_act)
class HelmholtzYKDD(base_class.YKDD):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.act = act
    self.k = kwargs['k']

  def Lnet_x_single(self, x: Tensor, theta) -> Tensor:
    return -dd_act(x @ theta['w'] + theta['b']) * (theta['w'] * theta['w']).sum(dim=0) - self.k**2 * act(x @ theta['w'] + theta['b'])
  
  def fluxnet_x_single(self, x: Tensor, theta, flux_info: Tensor) -> Tensor:
    return d_act(x @ theta['w'] + theta['b']) * torch.einsum('dM, d->M', theta['w'], flux_info)

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
  
  def sin_prob(k:float, a: float, b: float):
    target_exact = lambda x: (pi * a * x[...,0]).sin() * (pi * b * x[...,1]).sin()
    target_exact_grad = base_class.act_last(jacrev(target_exact), 0, chunk_size=chunk_size)
    target_rhs = lambda x: (-k**2 + pi**2 * (a**2 + b**2)) * (pi * a * x[...,0]).sin() * (pi * b * x[...,1]).sin()
    target_bou = lambda x: (pi * a * x[...,0]).sin() * (pi * b * x[...,1]).sin()
    return target_rhs, target_bou, target_exact, target_exact_grad
  
  u_fem_slice, n_fem = None, None
  fenicsx_path = Path('fem_sols')

  if problem == 'sin1pi4pi':
    k = 5
    target_rhs, target_bou, target_exact, target_exact_grad = sin_prob(k, 1, 4)

  return target_rhs, target_bou, target_exact, target_exact_grad, k
  
if __name__ == '__main__':
  t1 = time.time()
  target_rhs, target_bou, target_exact, target_exact_grad, k = get_problem(args.problem)
  
  ykdd = HelmholtzYKDD(
    target_rhs = target_rhs,
    target_bou = target_bou,
    k = k
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
  if args.solver in {'neum', 'table_theta'}:
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

      uses_fem_ref = args.problem in {'grf_16_0b', 'grf_32_0b'}
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
      new_cmap = truncate_colormap(cmap, 0., 1., N=1024, gamma=.6)
      log_norm = lambda y: mpl.colors.LogNorm(vmin=y.min(), vmax=y.max())

      savefig = lambda title: plt.savefig(f'figures/helmholtz{k}_{args.problem}_{N}_{args.plot_n}_{title}.{save_format}', format=save_format, bbox_inches='tight')

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