from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import torch
from typing import Callable
from functools import partial

Tensor = torch.Tensor
dtype = torch.float64
ones = partial(torch.ones, dtype=dtype)
zeros = partial(torch.zeros, dtype=dtype)
empty = partial(torch.empty, dtype=dtype)

def cg_distributed(
  multiplicity: Tensor,
  problem_size: int,
  A: Callable[[Tensor], Tensor],
  b: Tensor,
  x0: Tensor=None,
  *,
  M: Callable[[Tensor], Tensor]=lambda x: x,
  tol: float=1e-5,
  atol: float=0,
  maxiter: int=None,
  ):
  if x0 is None:
    x0 = zeros(b.shape)
  if maxiter is None:
    maxiter = problem_size * 3
    
  rsold, rsnew, pAp = empty([]), empty([]), empty([])

  x = x0
  r = b - A(x)
  z = M(r)
  p = z
  
  comm.Allreduce(r @ (z / multiplicity), rsold, op=MPI.SUM)

  atol = torch.maximum(tol * rsold ** .5, torch.tensor(atol * problem_size ** .5, dtype=dtype))

  i = 0
  while i < maxiter and rsold ** .5 > atol:
    Ap = A(p)
    comm.Allreduce(p @ (Ap / multiplicity), pAp, op=MPI.SUM)
    alpha = rsold / pAp
    x = x + alpha * p
    r = r - alpha * Ap
    z = M(r)
    comm.Allreduce(r @ (z / multiplicity), rsnew, op=MPI.SUM)
    beta = rsnew / rsold
    p = z + beta * p
    rsold[None] = rsnew
    i += 1

  return x, (i, rsold ** .5)

# weird intel mpi internode bug with Allreduce
# Iallreduce into Bcast also works, but is marginally slower
def cg_distributed_packeted(
  multiplicity: Tensor,
  problem_size: int,
  A: Callable[[Tensor], Tensor],
  b: Tensor,
  x0: Tensor=None,
  *,
  M: Callable[[Tensor], Tensor]=lambda x: x,
  tol: float=1e-5,
  atol: float=0,
  maxiter: int=None,
  ):
  if x0 is None:
    x0 = zeros(b.shape)
  if maxiter is None:
    maxiter = problem_size * 3
    
  buffer_size = 64
  rsold, rsnew, pAp = empty(buffer_size), empty(buffer_size), empty(buffer_size)
  rsold_aux, rsnew_aux, pAp_aux = empty(buffer_size), empty(buffer_size), empty(buffer_size)

  x = x0
  r = b - A(x)
  z = M(r)
  p = z
  
  rsold_aux[0] = r @ (z / multiplicity)
  comm.Allreduce(rsold_aux, rsold, op=MPI.SUM)

  atol = torch.maximum(tol * rsold[0] ** .5, torch.tensor(atol * problem_size ** .5, dtype=dtype))

  i = 0
  while i < maxiter and rsold[0] ** .5 > atol:
    Ap = A(p)
    pAp_aux[0] = p @ (Ap / multiplicity)
    comm.Allreduce(pAp_aux, pAp, op=MPI.SUM)
    alpha = rsold[0] / pAp[0]
    x = x + alpha * p
    r = r - alpha * Ap
    z = M(r)
    rsnew_aux[0] = r @ (z / multiplicity)
    comm.Allreduce(rsnew_aux, rsnew, op=MPI.SUM)
    beta = rsnew[0] / rsold[0]
    p = z + beta * p
    rsold[0] = rsnew[0]
    i += 1

  return x, (i, rsold[0] ** .5)

def cg_distributed_cond(
  multiplicity: Tensor,
  problem_size: int,
  A: Callable[[Tensor], Tensor],
  b: Tensor,
  x0: Tensor=None,
  *,
  M: Callable[[Tensor], Tensor]=lambda x: x,
  tol: float=1e-5,
  atol: float=0,
  maxiter: int=None,
  ):
  problem_size = empty([])
  comm.Allreduce((1 / multiplicity).sum(), problem_size, op=MPI.SUM)
  problem_size = int(problem_size.item())

  if x0 is None:
    x0 = zeros(b.shape)
  if maxiter is None:
    maxiter = problem_size * 10
    
  rsold, rsnew, pAp = empty([]), empty([]), empty([])

  x = x0
  r = b - A(x)
  z = M(r)
  p = z
  
  comm.Allreduce(r @ (z / multiplicity), rsold, op=MPI.SUM)

  atol = torch.maximum(tol * rsold ** .5, torch.tensor(atol * problem_size ** .5, dtype=dtype))

  if rank == 0:
    alpha = empty(maxiter)
    beta = empty(maxiter)
    i = 0
    while i < maxiter and rsold ** .5 > atol:
      Ap = A(p)
      comm.Allreduce(p @ (Ap / multiplicity), pAp, op=MPI.SUM)
      alpha[i] = rsold / pAp
      x = x + alpha[i] * p
      r = r - alpha[i] * Ap
      z = M(r)
      comm.Allreduce(r @ (z / multiplicity), rsnew, op=MPI.SUM)
      beta[i] = rsnew / rsold
      p = z + beta[i] * p
      rsold[None] = rsnew
      i += 1
    
    H = zeros(i, i)
    torch.diagonal(H, 0)[:] = 1 / alpha[:i]
    torch.diagonal(H, 0)[1:] += beta[:i-1] / alpha[:i-1]
    torch.diagonal(H, 1)[:] = -beta[:i-1]** .5 / alpha[:i-1]
    torch.diagonal(H, -1)[:] = torch.diagonal(H, 1)[:]
    print(f'approx cond: {torch.linalg.cond(H):.4e}')
  else:
    i = 0
    while i < maxiter and rsold ** .5 > atol:
      Ap = A(p)
      comm.Allreduce(p @ (Ap / multiplicity), pAp, op=MPI.SUM)
      alpha = rsold / pAp
      x = x + alpha * p
      r = r - alpha * Ap
      z = M(r)
      comm.Allreduce(r @ (z / multiplicity), rsnew, op=MPI.SUM)
      beta = rsnew / rsold
      p = z + beta * p
      rsold[None] = rsnew
      i += 1

  return x, (i, rsold ** .5)

def gmres_distributed(
  multiplicity: Tensor,
  problem_size: int,
  A: Callable[[Tensor], Tensor],
  b: Tensor,
  x0: Tensor=None,
  *,
  M: Callable[[Tensor], Tensor]=lambda x: x,
  tol: float=1e-5,
  atol: float=0,
  maxiter: int=None,
  restart: int=20,
  safety_eps: float=1e-40,
  mgs: bool=False,
  ):
  if maxiter is None:
    maxiter = problem_size * 3
  if restart is None:
    restart = problem_size
  if restart > problem_size:
    restart = problem_size

  Q = empty(b.shape[0], restart)
  H = zeros(restart + 1, restart)
  H_gs = zeros(restart)

  r0_ss, v_ss = empty([]), empty([])

  if x0 is None:
    x0 = zeros(b.shape)
    r0 = b
    comm.Allreduce(r0 @ (r0 / multiplicity), r0_ss, op=MPI.SUM)
  else:
    r0 = b - M(A(x0))
    comm.Allreduce(r0 @ (r0 / multiplicity), r0_ss, op=MPI.SUM)
  atol = torch.maximum(tol * r0_ss ** .5, torch.tensor(atol * problem_size ** .5, dtype=dtype))

  e1 = zeros(restart + 1)
  e1[0] = 1
  first_flag = True
  for i_iter in range(maxiter):
    p = ones(1) # the orthogonal complement of H

    if first_flag:
      first_flag = False
      v_ss[None] = r0_ss
    else:
      r0 = b - M(A(x0))
      comm.Allreduce(r0 @ (r0 / multiplicity), v_ss, op=MPI.SUM)

    v = r0
    v_l2 = v_ss ** .5
    r0_l2 = v_l2
    for i_restart in range(restart):
      Q[:, i_restart] = v / v_l2
      v = M(A(Q[:, i_restart]))

      if mgs:
        # modified Gram Schmidt
        for i_mgs in range(i_restart + 1):
          comm.Allreduce(Q[:, i_mgs] @ (v / multiplicity), H[i_mgs, i_restart], op=MPI.SUM)
          v -= Q[:, i_mgs] * H[i_mgs, i_restart]

      else:
        # no mod
        comm.Allreduce((v / multiplicity) @ Q[:, :i_restart+1], H_gs[:i_restart+1], op=MPI.SUM)
        H[:i_restart+1, i_restart] = H_gs[:i_restart+1]
        v -= Q[:, :i_restart+1] @ H[:i_restart+1, i_restart]

      comm.Allreduce(v @ (v / multiplicity), v_ss, op=MPI.SUM)
      v_l2 = v_ss ** .5
      H[i_restart + 1, i_restart] = v_l2

      w = p @ H[:i_restart+1, i_restart]
      p = torch.cat([v_l2 * p, -w.unsqueeze(0)])
      # p = p / (p @ p).sqrt()
      p = p / (H[i_restart+1, i_restart] ** 2 + w ** 2) ** .5
      
      # res = (e1[:i_restart+2] @ p) * r0_l2
      res = p[0] * r0_l2
      if res < atol:
        break
    
    e1_approx = r0_l2 * e1[:i_restart+2] - res * p
    if H[i_restart+1, i_restart] == 0:
      H[i_restart+1, i_restart] = safety_eps
    y = torch.linalg.solve_triangular(H[1:i_restart+2, :i_restart+1], e1_approx[1:].unsqueeze(1), upper=True).reshape(-1)
    # y = torch.linalg.lstsq(H[:i_restart+2, :i_restart+1], e1_approx, driver='gelsd').solution

    x0 += torch.einsum('ji, i->j', Q[:, :i_restart+1], y)
    if res < atol:
      break
    
  return x0, (i_iter+1, i_restart+1, res)

def gmres_lanczos_distributed(
  multiplicity: Tensor,
  problem_size: int,
  A: Callable[[Tensor], Tensor],
  b: Tensor,
  x0: Tensor=None,
  *,
  M: Callable[[Tensor], Tensor]=None,
  tol: float=1e-5,
  atol: float=0,
  maxiter: int=None,
  restart: int=None,
  safety_eps: float=1e-40,
  ):
  if maxiter is None:
    maxiter = 1
  if restart is None:
    restart = problem_size * 3
  # if restart > problem_size:
  #   restart = problem_size

  MA = (lambda x: M(A(x))) if M is not None else A

  r0_ss, v_ss = empty([]), empty([])

  if x0 is None:
    x0 = zeros(b.shape)
    r0 = b
    comm.Allreduce(r0 @ (r0 / multiplicity), r0_ss, op=MPI.SUM)
  else:
    r0 = b - MA(x0)
    comm.Allreduce(r0 @ (r0 / multiplicity), r0_ss, op=MPI.SUM)
  atol = torch.maximum(tol * r0_ss ** .5, torch.tensor(atol * problem_size ** .5, dtype=dtype))
  
  first_flag = True
  for i_iter in range(maxiter):
    if first_flag:
      first_flag = False
      v_ss[None] = r0_ss
    else:
      r0 = b - MA(x0)
      comm.Allreduce(r0 @ (r0 / multiplicity), v_ss, op=MPI.SUM)

    h_k_km1, h_k_k = 0, zeros([])
    q_km1 = zeros(b.shape)

    v = r0
    v_l2 = v_ss ** .5
    r0_l2 = v_l2
    q_k = v / v_l2

    c_km2, s_km2, c_km1, s_km1 = 1, 0, -1, 0
    res_km1 = r0_l2
    q_km2_hat, q_km1_hat = zeros(b.shape), zeros(b.shape)

    x_km1 = zeros(b.shape)
    for i_restart in range(restart):
      v = MA(q_k)
      Aq_k = v
      h_km1_k = h_k_km1
      v = v - h_km1_k * q_km1
      
      comm.Allreduce(q_k @ (v / multiplicity), h_k_k, op=MPI.SUM)

      v = v - h_k_k * q_k

      comm.Allreduce(v @ (v / multiplicity), v_ss, op=MPI.SUM)
      v_l2 = v_ss ** .5
      h_kp1_k = v_l2

      q_kp1 = v / v_l2

      w_k = -c_km2 * s_km1 * h_km1_k - c_km1 * h_k_k
      n_k = (w_k ** 2 + h_kp1_k ** 2) ** .5
      c_k, s_k = w_k / n_k, h_kp1_k / n_k

      res_k = s_k * res_km1

      q_k_hat = (q_k - s_km2 * h_km1_k * q_km2_hat + (c_km2 * c_km1 * h_km1_k - s_km1 * h_k_k) * q_km1_hat) / n_k

      x_k = x_km1 + c_k * res_km1 * q_k_hat
  
      if res_k < atol:
        break

      h_k_km1 = h_kp1_k
      q_km1, q_k = q_k, q_kp1

      c_km2, s_km2, c_km1, s_km1 = c_km1, s_km1, c_k, s_k
      res_km1 = res_k
      q_km2_hat, q_km1_hat = q_km1_hat, q_k_hat

      x_km1 = x_k
    
    x0 = x_k

    if res_k < atol:
      break
    
  return x0, (i_iter+1, i_restart+1, res_k)

def get_linear_solver(name: str, *, tol=1e-12, atol=0, maxiter=None, restart=None, mgs=False) -> Callable:
  if name == 'cg':
    return partial(cg_distributed, tol=tol, atol=atol, maxiter=maxiter)
  elif name == 'cg_packeted':
    return partial(cg_distributed_packeted, tol=tol, atol=atol, maxiter=maxiter)
  elif name == 'cg_cond':
    return partial(cg_distributed_cond, tol=tol, atol=atol, maxiter=maxiter)
  elif name == 'gmres':
    return partial(gmres_distributed, tol=tol, atol=atol, maxiter=maxiter, restart=restart, mgs=mgs)
  elif name == 'gmres_lanczos':
    return partial(gmres_lanczos_distributed, tol=tol, atol=atol, maxiter=maxiter, restart=restart)
  else:
    raise ValueError(f'Unknown linear solver: {name}')