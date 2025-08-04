import torch
from typing import Callable, Union, NamedTuple, Any
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
eye = partial(torch.eye, dtype=dtype)

import runtime_constants
chunk_size = runtime_constants.chunk_size

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from linear_solvers import cg_distributed

# basically direct lift of answer in stackoverflow
# https://stackoverflow.com/questions/11065419/unpacking-keyword-arguments-but-only-the-ones-that-match-the-function
def nonkwargs_to_kwargs(foo):
  if any([arg.kind == arg.VAR_KEYWORD for arg in signature(foo).parameters.values()]):
    return foo
  @wraps(foo)
  def wrapped_foo(*args, **kwargs):
    subset = dict((key, kwargs[key]) for key in kwargs if key in signature(foo).parameters)
    return foo(*args, **subset)
  return wrapped_foo

def act_last(f: Callable, where_x: Union[int, list[int]]=0, chunk_size: int=None):
  in_dims = [None for v in signature(f).parameters.values() if v.kind is v.POSITIONAL_OR_KEYWORD]
  if isinstance(where_x, int):
    in_dims[where_x] = 0
  else:
    for i in where_x:
      in_dims[i] = 0
  in_dims = tuple(in_dims)
  vfoo = vmap(f, in_dims=in_dims, chunk_size=chunk_size)
  @wraps(f)
  def wrapper(*args, **kwargs):
    if isinstance(where_x, int):
      shape = args[where_x].shape
      args = list(args)
      args[where_x] = args[where_x].reshape(-1, shape[-1])
    else:
      shape = args[where_x[0]].shape
      args = list(args)
      for i in where_x:
        args[i] = args[i].reshape(-1, shape[-1])
    rax = vfoo(*args, **kwargs)
    return rax.reshape(shape[:-1] + rax.shape[1:])
  return wrapper

class Interface(NamedTuple):
  idcs: Tensor
  rank: int
  flux_info: Tensor
  send_tag: int=0
  recv_tag: int=0

class Grid(NamedTuple):
  dual_size: int
  prim_size: int
  int: Tensor
  bou: Tensor
  dual: Tensor
  prim: Tensor
  dual_intfc: list[Interface]
  dual_to_glob: Tensor
  dual_multiplicity: Tensor
  prim_intfc: list[Interface]
  prim_to_glob: Tensor
  prim_multiplicity: Tensor
  bou_to_dual: Tensor=None
  bou_to_prim: Tensor=None
  n: Any=None

class YKDD(ABC):
  def __init__(self, **kwargs) -> None:
    self.act: Callable[[Tensor], Tensor] = torch.tanh

    self.define_problem_params(**kwargs)

  @nonkwargs_to_kwargs
  def define_problem_params(
    self, *,
    target_rhs: Callable[[Tensor], Tensor]=None,
    target_bou: Callable[[Tensor], Tensor]=None
  ) -> None:
    self.target_rhs = target_rhs
    self.target_bou = target_bou

  def net_x(self, x: Tensor, theta) -> Tensor:
    return self.act(x @ theta['w'] + theta['b'])
  
  def net_xa(self, x: Tensor, a: Tensor, theta) -> Tensor:
    return self.net_x(x, theta) @ a
  
  @partial(act_last, where_x=1, chunk_size=chunk_size)
  def dx_net_x(self, x: Tensor, theta) -> Tensor:
    return jacfwd(self.net_x, argnums=0)(x, theta)
  
  def dx_net_xa(self, x: Tensor, a: Tensor, theta) -> Tensor:
    return torch.einsum('...Md, M->...d', self.dx_net_x(x, theta), a)
  
  @abstractmethod
  def Lnet_x_single(self, x: Tensor, theta) -> Tensor:
    return empty(x.shape[:-1])

  @partial(act_last, where_x=1, chunk_size=chunk_size)
  def Lnet_x(self, x: Tensor, theta) -> Tensor:
    return self.Lnet_x_single(x, theta)

  def Lnet_xa(self, x: Tensor, a: Tensor, theta) -> Tensor:
    return self.Lnet_x(x, theta) @ a
  
  @abstractmethod
  def fluxnet_x_single(self, x: Tensor, theta, flux_info: Tensor) -> Tensor:
    return empty(x.shape[:-1])
  
  @partial(act_last, where_x=[1, 3], chunk_size=chunk_size)
  def fluxnet_x(self, x: Tensor, theta, flux_info: Tensor) -> Tensor:
    return self.fluxnet_x_single(x, theta, flux_info)

  def get_grid(self, *, n):
    """Points for domain decomposition in standard rectangular 2D grid."""
    N = int(size ** .5)
    i, j = rank // N, rank % N
    linx = linspace(0, 1, n+1) / N
    grid = torch.cartesian_prod(linx + i / N, linx + j / N).reshape(n+1, n+1, 2)

    x_int = grid[1:-1, 1:-1].reshape(-1, 2)
    # x_int = torch.rand((n-1)**2, 2, dtype=dtype) / N + torch.tensor([i/N, j/N], dtype=dtype)
    x_bou = torch.cat(
      [
        grid[0, :-1],
        grid[:-1, -1],
        grid[-1, 1:].flip(0),
        grid[1:, 0].flip(0)
      ]
    )

    empty_idx = torch.arange(0)
    ind_bou_idcs = torch.unique(
      torch.cat(
        [
          torch.arange(n+1) if i == 0 else empty_idx,
          torch.arange(n, 2*n+1) if j == N-1 else empty_idx,
          torch.arange(2*n, 3*n+1) if i == N-1 else empty_idx,
          torch.arange(3*n, 4*n) if j == 0 else empty_idx,
          torch.arange(1) if j == 0 else empty_idx
        ]
      )
    )
    x_ind_bou = x_bou[ind_bou_idcs]

    gam_mask = torch.ones(4*n, dtype=torch.bool)
    gam_mask[ind_bou_idcs] = False
    dual_mask = gam_mask.clone()
    dual_mask[torch.arange(0, 4*n, n)] = False
    bou_to_dual = dual_mask.cumsum(0) - 1
    x_dual = x_bou[dual_mask]
    prim_mask = torch.logical_and(gam_mask, torch.logical_not(dual_mask))
    bou_to_prim = prim_mask.cumsum(0) - 1
    x_prim = x_bou[prim_mask]

    dual_interface = []
    if i > 0:
      # up
      idcs_to_x_bou = torch.arange(1, n)
      dual_interface.append(
        Interface(
          bou_to_dual[idcs_to_x_bou], # idcs to x_gam
          rank - N, # rank sharing these points
          torch.tensor([-1, 0], dtype=dtype).expand(idcs_to_x_bou.shape[0], 2) # information to get flux; in this case the normal vector
        )
      )
    if j < N-1:
      # right
      idcs_to_x_bou = torch.arange(n+1, 2*n)
      dual_interface.append(
        Interface(
          bou_to_dual[idcs_to_x_bou],
          rank + 1,
          torch.tensor([0, 1], dtype=dtype).expand(idcs_to_x_bou.shape[0], 2)
        )
      )
    if i < N-1:
      # down
      idcs_to_x_bou = torch.arange(3*n-1, 2*n, -1)
      dual_interface.append(
        Interface(
          bou_to_dual[idcs_to_x_bou], # flip direction to match neighbor's up
          rank + N,
          torch.tensor([1, 0], dtype=dtype).expand(idcs_to_x_bou.shape[0], 2)
        )
      )
    if j > 0:
      # left
      idcs_to_x_bou = torch.arange(-1, -n, -1)
      dual_interface.append(
        Interface(
          bou_to_dual[idcs_to_x_bou],
          rank - 1,
          torch.tensor([0, -1], dtype=dtype).expand(idcs_to_x_bou.shape[0], 2)
        )
      )

    prim_interface = []
    # corners
    # up left; sends tag 0
    if i > 0 and j > 0:
      prim_interface.append(Interface(bou_to_prim[0:1], rank - N - 1, None, 0, 2))
      prim_interface.append(Interface(bou_to_prim[0:1], rank - N, torch.tensor([-1, 0], dtype=dtype).expand(1, 2), 0, 3))
      prim_interface.append(Interface(bou_to_prim[0:1], rank - 1, torch.tensor([0, -1], dtype=dtype).expand(1, 2), 0, 1))
    # up right; sends tag 1
    if i > 0 and j < N-1:
      prim_interface.append(Interface(bou_to_prim[n:n+1], rank - N + 1, None, 1, 3))
      prim_interface.append(Interface(bou_to_prim[n:n+1], rank - N, torch.tensor([-1, 0], dtype=dtype).expand(1, 2), 1, 2))
      prim_interface.append(Interface(bou_to_prim[n:n+1], rank + 1, torch.tensor([0, 1], dtype=dtype).expand(1, 2), 1, 0))
    # down right; sends tag 2
    if i < N-1 and j < N-1:
      prim_interface.append(Interface(bou_to_prim[2*n:2*n+1], rank + N + 1, None, 2, 0))
      prim_interface.append(Interface(bou_to_prim[2*n:2*n+1], rank + N, torch.tensor([1, 0], dtype=dtype).expand(1, 2), 2, 1))
      prim_interface.append(Interface(bou_to_prim[2*n:2*n+1], rank + 1, torch.tensor([0, 1], dtype=dtype).expand(1, 2), 2, 3))
    # down left; sends tag 3
    if i < N-1 and j > 0:
      prim_interface.append(Interface(bou_to_prim[3*n:3*n+1], rank + N - 1, None, 3, 1))
      prim_interface.append(Interface(bou_to_prim[3*n:3*n+1], rank + N, torch.tensor([1, 0], dtype=dtype).expand(1, 2), 3, 0))
      prim_interface.append(Interface(bou_to_prim[3*n:3*n+1], rank - 1, torch.tensor([0, -1], dtype=dtype).expand(1, 2), 3, 2))

    #   y0  ------------------->  1
    #   -----------------------------
    # x |  0   |  1   |  2   |  3   |
    # 0 |      0      1      2      |
    #   |--12--o--15--o--18--o--21--|
    # | |  4   |  5   |  6   |  7   |  
    # | |      3      4      5      |
    # | |--13--o--16--o--19--o--22--|
    # | |  8   |  9   |  10  |  11  |
    # | |      6      7      8      |
    # V |--14--o--17--o--20--o--23--|
    #   |  12  |  13  |  14  |  15  |
    # 1 |      9     10     11      |
    #   -----------------------------
    # the o are numbered
    # 24  25  26
    # 27  28  29
    # 30  31  32

    one_idx = torch.arange(1,2)
    dual_to_glob = torch.cat(
      [
        torch.arange(0, n-1) + (n-1)*(N*(N-1) + (N-1)*j + i-1) if i > 0 else empty_idx, # up
        torch.arange(0, n-1) + (n-1)*((N-1)*i + j) if j < N-1 else empty_idx, # right
        torch.arange(n-2, -1, -1) + (n-1)*(N*(N-1) + (N-1)*j + i) if i < N-1 else empty_idx, # down, flip direction
        torch.arange(n-2, -1, -1) + (n-1)*((N-1)*i + j-1) if j > 0 else empty_idx, # left
      ]
    )

    prim_to_glob = torch.cat(
      [
        one_idx * ((N-1)*(i-1) + j-1) if i > 0 and j > 0 else empty_idx, # up left corner
        one_idx * ((N-1)*(i-1) + j) if i > 0 and j < N-1 else empty_idx, # up right corner
        one_idx * ((N-1)*i + j) if i < N-1 and j < N-1 else empty_idx, # down right corner
        one_idx * ((N-1)*i + j-1) if i < N-1 and j > 0 else empty_idx, # down left corner
      ]
    )

    dual_multiplicity = torch.cat(
      [
        ones(n-1) * 2 if i > 0 else empty_idx, # up
        ones(n-1) * 2 if j < N-1 else empty_idx, # right
        ones(n-1) * 2 if i < N-1 else empty_idx, # down
        ones(n-1) * 2 if j > 0 else empty_idx, # left
      ]
    )

    prim_multiplicity = torch.cat(
      [
        one_idx * 4 if i > 0 and j > 0 else empty_idx, # up left corner
        one_idx * 4 if i > 0 and j < N-1 else empty_idx, # up right corner
        one_idx * 4 if i < N-1 and j < N-1 else empty_idx, # down right corner
        one_idx * 4 if i < N-1 and j > 0 else empty_idx, # down left corner
      ]
    )

    return Grid(
      2*N*(N-1)*(n-1), (N-1)**2,
      x_int, x_ind_bou, x_dual, x_prim,
      dual_interface, dual_to_glob, dual_multiplicity,
      prim_interface, prim_to_glob, prim_multiplicity,
      bou_to_dual=bou_to_dual, bou_to_prim=bou_to_prim, n=n
    )
  
  def get_K_o(self, grid: Grid, theta) -> tuple[Tensor, tuple[int, int], Callable[[Tensor], Tensor]]:
    K = torch.cat(
      [
        self.Lnet_x(grid.int, theta),
        self.net_x(grid.bou, theta),
        self.net_x(grid.prim, theta),
        self.net_x(grid.dual, theta)
      ]
    )

    def prim_idcs_to_dof_idcs(idcs: Tensor) -> Tensor:
      return idcs
    
    def dual_idcs_to_dof_idcs(idcs: Tensor) -> Tensor:
      return idcs

    def glob_prim_idcs_to_glob_dof_idcs(idcs: Tensor) -> Tensor:
      return idcs
    
    return K, (grid.prim.shape[0], grid.dual.shape[0]), (prim_idcs_to_dof_idcs, dual_idcs_to_dof_idcs, glob_prim_idcs_to_glob_dof_idcs)
  
  def get_K(self, grid: Grid, theta) -> tuple[Tensor, tuple[int, int], Callable[[Tensor], Tensor]]:
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

    K = torch.cat(
      [
        self.Lnet_x(grid.int, theta),
        self.net_x(grid.bou, theta),
        K_prim,
        K_dual
      ]
    )

    def prim_idcs_to_dof_idcs(idcs: Tensor) -> Tensor:
      return idcs
    
    def dual_idcs_to_dof_idcs(idcs: Tensor) -> Tensor:
      return idcs

    def glob_prim_idcs_to_glob_dof_idcs(idcs: Tensor) -> Tensor:
      return idcs
    
    return K, (grid.prim.shape[0], grid.dual.shape[0]), (prim_idcs_to_dof_idcs, dual_idcs_to_dof_idcs, glob_prim_idcs_to_glob_dof_idcs)

  def get_rhs(self, grid: Grid) -> Tensor:
    return torch.cat(
      [
        self.target_rhs(grid.int),
        self.target_bou(grid.bou),
        zeros(grid.prim.shape[0]),
        zeros(grid.dual.shape[0])
      ]
    )
  
  def get_A(self, grid: Grid, theta) -> tuple[Tensor, list[tuple[int, int]], Tensor, list[tuple[int, int]]]:
    idx = 0
    idcs_prim = []
    empty_flux = self.empty_flux(theta)
    A_prim = [empty_flux]
    for interface in grid.prim_intfc:
      if interface.flux_info is not None:
        A_prim.append(self.fluxnet_x(grid.prim[interface.idcs], theta, interface.flux_info))
      else:
        A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]

    idx = 0
    idcs_dual = []
    A_dual = [empty_flux]
    for interface in grid.dual_intfc:
      if interface.flux_info is not None:
        A_dual.append(self.fluxnet_x(grid.dual[interface.idcs], theta, interface.flux_info))
      else:
        A_dual.append(empty_flux)
      idcs_dual.append((idx, idx + A_dual[-1].shape[0]))
      idx = idcs_dual[-1][1]

    return torch.cat(A_prim), idcs_prim, torch.cat(A_dual), idcs_dual
  
  def get_A_o(self, grid: Grid, theta) -> tuple[Tensor, list[tuple[int, int]], Tensor, list[tuple[int, int]]]:
    idx = 0
    idcs_prim = []
    empty_flux = self.empty_flux(theta)
    A_prim = [empty_flux]
    for interface in grid.prim_intfc:
      if interface.flux_info is not None:
        A_prim.append(self.fluxnet_x(grid.prim[interface.idcs], theta, interface.flux_info))
      else:
        A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]

    idx = 0
    idcs_dual = []
    A_dual = [empty_flux]
    for interface in grid.dual_intfc:
      if interface.flux_info is not None:
        A_dual.append(self.fluxnet_x(grid.dual[interface.idcs], theta, interface.flux_info))
      else:
        A_dual.append(empty_flux)
      idcs_dual.append((idx, idx + A_dual[-1].shape[0]))
      idx = idcs_dual[-1][1]

    A_prim = torch.cat(A_prim)
    A_dual = torch.cat(A_dual)

    return A_prim, idcs_prim, A_dual, idcs_dual
  
  def empty_flux(self, theta) -> Tensor:
    return empty(0, theta['b'].shape[0])

  def get_A(self, grid: Grid, theta) -> tuple[Tensor, list[tuple[int, int]], Tensor, list[tuple[int, int]]]:
    N = int(size ** .5)
    i, j = rank // N, rank % N
    n = grid.n
    bou_to_dual = grid.bou_to_dual
    bou_to_prim = grid.bou_to_prim

    idx = 0
    idcs_prim = []
    empty_flux = self.empty_flux(theta)
    A_prim = [empty_flux]
    n_for_avg = n * 1
    if i > 0 and j > 0:
      A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append((
        self.fluxnet_x(grid.prim[bou_to_prim[0:1]], theta, torch.tensor([-1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[1:n]], theta, torch.tensor([-1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      )/(n_for_avg))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append((
        self.fluxnet_x(grid.prim[bou_to_prim[0:1]], theta, torch.tensor([0, -1], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[3*n+1:4*n]], theta, torch.tensor([0, -1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      )/(n_for_avg))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
    # up right; sends tag 1
    if i > 0 and j < N-1:
      A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append((
        self.fluxnet_x(grid.prim[bou_to_prim[n:n+1]], theta, torch.tensor([-1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[1:n]], theta, torch.tensor([-1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      )/(n_for_avg))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append((
        self.fluxnet_x(grid.prim[bou_to_prim[n:n+1]], theta, torch.tensor([0, 1], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[n+1:2*n]], theta, torch.tensor([0, 1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      )/(n_for_avg))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
    # down right; sends tag 2
    if i < N-1 and j < N-1:
      A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append((
        self.fluxnet_x(grid.prim[bou_to_prim[2*n:2*n+1]], theta, torch.tensor([1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[2*n+1:3*n]], theta, torch.tensor([1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      )/(n_for_avg))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append((
        self.fluxnet_x(grid.prim[bou_to_prim[2*n:2*n+1]], theta, torch.tensor([0, 1], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[n+1:2*n]], theta, torch.tensor([0, 1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      )/(n_for_avg))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
    # down left; sends tag 3
    if i < N-1 and j > 0:
      A_prim.append(empty_flux)
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append((
        self.fluxnet_x(grid.prim[bou_to_prim[3*n:3*n+1]], theta, torch.tensor([1, 0], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[2*n+1:3*n]], theta, torch.tensor([1, 0], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      )/(n_for_avg))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]
      A_prim.append((
        self.fluxnet_x(grid.prim[bou_to_prim[3*n:3*n+1]], theta, torch.tensor([0, -1], dtype=dtype).expand(1, 2)) +
        self.fluxnet_x(grid.dual[bou_to_dual[3*n+1:4*n]], theta, torch.tensor([0, -1], dtype=dtype).expand(n-1, 2)).sum(0, keepdims=True)
      )/(n_for_avg))
      idcs_prim.append((idx, idx + A_prim[-1].shape[0]))
      idx = idcs_prim[-1][1]

    idx = 0
    idcs_dual = []
    A_dual = [empty_flux]
    for interface in grid.dual_intfc:
      if interface.flux_info is not None:
        A_dual.append(self.fluxnet_x(grid.dual[interface.idcs], theta, interface.flux_info))
      else:
        A_dual.append(empty_flux)
      idcs_dual.append((idx, idx + A_dual[-1].shape[0]))
      idx = idcs_dual[-1][1]

    A_prim = torch.cat(A_prim)
    A_dual = torch.cat(A_dual)

    return A_prim, idcs_prim, A_dual, idcs_dual

  def get_multiplicity(self, grid: Grid) -> Tensor:
    return grid.prim_multiplicity, grid.dual_multiplicity
  
  # blocking communication for now
  def grid_communication(self, grid: Grid, message: list[Tensor], postbox: list[Tensor], prim: bool):
    send_reqs = []
    recv_reqs = []
    for (interface, m, p) in zip(grid.prim_intfc if prim else grid.dual_intfc, message, postbox):
      send_reqs.append(comm.Isend(m, dest=interface.rank, tag=interface.send_tag))
      recv_reqs.append(comm.Irecv(p, source=interface.rank, tag=interface.recv_tag))
    # while recv_reqs:
    #   for i, req in enumerate(recv_reqs):
    #     if req.Test():
    #       recv_reqs.pop(i)
    #       break
    for req in recv_reqs:
      req.wait()
    
    for req in send_reqs:
      req.wait()

    return
  
  def A_comms(self, grid: Grid, A: Tensor, A_idcs: list[tuple[int, int]], prim: bool) -> Tensor:
    A_recv = empty(A.shape)
    message = [A[i:j] for i, j in A_idcs]
    postbox = [A_recv[i:j] for i, j in A_idcs]
    self.grid_communication(grid, message, postbox, prim)
    return A + A_recv
  
  def BT_comms(self, grid: Grid, rbx: Tensor, BT_idcs: list[Tensor], prim: bool) -> Tensor:
    message = [rbx[idcs] for idcs in BT_idcs]
    postbox = [empty(m.shape) for m in message]
    self.grid_communication(grid, message, postbox, prim)
    rax = rbx.clone()
    for post, idcs in zip(postbox, BT_idcs):
      rax[idcs] += post
    return rax
  
  def ykdd(
      self, grid: Grid, theta, *,
      driver='gelsd', tol=1e-9, atol=0,
      lin_solver='cg', maxiter=None,
      rcond_factor=1e-0
    ):
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
    mask = S >= torch.tensor(rcond, dtype=K.dtype) * S[0] * rcond_factor
    safe_idx = mask.sum()
    U, S, Vh = U[:, :safe_idx], S[:safe_idx], Vh[:safe_idx]
    S_inv = 1 / S

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
    
    def apply_mat(u: Tensor) -> Tensor:
      D_pdu = self.A_comms(grid, D_pd @ u, A_idcs_prim, prim=True)
      n, mu_glob, mu_prim = E(u, D_pdu)
      rax2 = ET(n)
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
    rhs2 = ET(rhs_n)
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
  
  def ykdd_neum(self, grid: Grid, theta, *, driver='gelsd', tol=1e-9, atol=0, lin_solver='cg', Theta=1, maxiter=None):
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
    A_tilde = K[K.shape[0]-K_dual_size:]
    B_tilde = zeros(K_tilde.shape[0], A_dual.shape[0])
    B_tilde[K.shape[0]-K_dual_size:] = -eye(A_dual.shape[0])

    U_tilde, S_tilde, Vh_tilde = torch.linalg.svd(K_tilde, full_matrices=False)
    rcond = torch.finfo(K_tilde.dtype).eps * max(*K_tilde.shape)
    mask = S_tilde >= torch.tensor(rcond, dtype=K_tilde.dtype) * S_tilde[0]
    safe_idx = mask.sum()
    U_tilde, S_tilde, Vh_tilde = U_tilde[:, :safe_idx], S_tilde[:safe_idx], Vh_tilde[:safe_idx]
    S_inv_tilde = 1 / S_tilde

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

  def ykdd_table_theta(
      self, grid: Grid, theta, *,
      driver='gelsd', tol=1e-9, atol=0, lin_solver='cg',
      Theta=1, maxiter=None,
      error_function: Callable=None
    ):
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
    A_tilde = K[K.shape[0]-K_dual_size:]
    B_tilde = zeros(K_tilde.shape[0], A_dual.shape[0])
    B_tilde[K.shape[0]-K_dual_size:] = -eye(A_dual.shape[0])

    U_tilde, S_tilde, Vh_tilde = torch.linalg.svd(K_tilde, full_matrices=False)
    rcond = torch.finfo(K_tilde.dtype).eps * max(*K_tilde.shape)
    mask = S_tilde >= torch.tensor(rcond, dtype=K_tilde.dtype) * S_tilde[0]
    safe_idx = mask.sum()
    U_tilde, S_tilde, Vh_tilde = U_tilde[:, :safe_idx], S_tilde[:safe_idx], Vh_tilde[:safe_idx]
    S_inv_tilde = 1 / S_tilde

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
    
    for Theta in [0., .5, .9, .99, .999, .9999, 1.]:
      c, info = ykdd_table_theta_sub(Theta)
      if rank == 0:
        print(f'Theta: {Theta} | iters: {info[0]: 5d} | residual: {info[1]:.4e}')
      error_function(c)
    
    return c, info

  def elm(self, grid: Grid, theta, *, driver='gelsd', numpy=False, **_):
    K, _, _ = self.get_K(grid, theta)
    f = self.get_rhs(grid)
    if not numpy:
      a = torch.linalg.lstsq(K, f, driver=driver).solution
    else:
      print('using np lstsq')
      import numpy as np
      a = np.linalg.lstsq(K.numpy(), f.numpy(), rcond=None)[0]
      a = torch.from_numpy(a)
    return a, (0, 0)

  def compute_errors(
      self, grid: Grid, a: Tensor, theta,
      target_exact: Callable[[Tensor], Tensor],
      target_exact_grad: Callable[[Tensor], Tensor],
      *,
      suppress=False, flush=False
    ):
    l2_points = torch.cat([grid.int, grid.bou, grid.prim, grid.dual])
    y = self.net_xa(l2_points, a, theta)
    y_exact = target_exact(l2_points)
    y_exact_ss = torch.square(y_exact).sum()
    y_error_ss = torch.square(y - y_exact).sum()

    h1_points = torch.cat([grid.int, grid.prim, grid.dual])
    y_grad = self.dx_net_xa(h1_points, a, theta)
    y_exact_grad = target_exact_grad(h1_points)
    y_exact_grad_ss = torch.square(y_exact_grad).sum()
    y_grad_error_ss = torch.square(y_grad - y_exact_grad).sum()

    mu_points = torch.cat([grid.prim, grid.dual])
    mu = self.net_xa(mu_points, a, theta)
    mu_exact = target_exact(mu_points)
    mu_error_ss = (torch.square(mu - mu_exact) / torch.cat([grid.prim_multiplicity, grid.dual_multiplicity])).sum()
    mu_n = (1 / grid.prim_multiplicity).sum() + (1 / grid.dual_multiplicity).sum()

    res = self.Lnet_xa(grid.int, a, theta) - self.target_rhs(grid.int)
    res_ss = torch.square(res).sum()
    res_n = grid.int.shape[0]

    packet = torch.tensor(
      [y_exact_ss, y_error_ss, y_exact_grad_ss, y_grad_error_ss, mu_error_ss, mu_n, res_ss, res_n], dtype=dtype
    )
    packet_recv = empty(packet.shape)
    comm.Reduce(packet, packet_recv, op=MPI.SUM, root=0)

    return self.print_error(packet_recv, suppress=suppress, flush=flush)

  def print_error(self, packet_recv, *, suppress=False, flush=False):
    if rank == 0:
      l2 = (packet_recv[1] / packet_recv[0])**.5
      h1 = ((packet_recv[1] + packet_recv[3]) / (packet_recv[0] + packet_recv[2]))**.5
      l2_mu = (packet_recv[4] / packet_recv[5])**.5
      l2_residual = (packet_recv[6] / packet_recv[7])**.5
      if not suppress: print(
        f'l2 error: {l2 : .4e}\n'
        f'h1 error: {h1 : .4e}\n'
        f'l2 error mu: {l2_mu : .4e}\n'
        f'l2 error residual: {l2_residual : .4e}',
        flush=flush
      )
      return l2, h1, l2_mu, l2_residual
    
    else:
      return None, None, None, None
