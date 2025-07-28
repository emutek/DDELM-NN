# DDELM-NN
A Neumann-Neumann Acceleration with Coarse Space for Domain Decomposition of Extreme Learning Machines

We implement domain decomposition algorithms for extreme learning machines from the following paper:

  - Byungeun Ryoo and Chang-Ock Lee. "[A Neumann-Neumann Acceleration with Coarse Space for Domain Decomposition of Extreme Learning Machines](https://arxiv.org/abs/2503.10032)." arXiv preprint arXiv:2503.10032 (2025).

## Dependencies
Version used during development noted in parantheses
  - Pytorch (2.2.2)
  - MPI for Python (4.0.0)

Visualization uses
  - Matplotlib (3.8.0)

## Usage

MPI command syntax will depend on the MPI implementation of your system.

Basic usage (spawning 64 subprocesses with 32 by 32 grid on each subdomain and 1024 neurons per subdomain solving the Poisson equation):
```
mpirun -n 64 python poisson.py --n 32 --M 1024
```

Number of subdomains spawned should be a square number.

### Some commands used for paper
```
mpirun -n 64 python poisson.py --n 32 --M 0x400 --error_n 64 --plot_n 63 --save_img_paper -p grf_32_0b
mpirun -n 64 python poisson_var.py --n 64 --M 0x1000 -error_n 128 --plot_n 127 --save_img_paper -p grf_32_0b_1f
mpirun -n 64 python plate_bending.py --n 32 --M 0x400 --error_n 64 --plot_n 63 --save_img_paper
```

### Options
  - `--n` subdomain grid will be $$(n+1)\times(n+1)$$
  - `--M` number of neurons assigned to each subdomain
  - `-p` designates the problem to solve; refer to each `poisson.py`, `poisson_var.py`, `plate_bending.py` for available problems
  - `--error_n` subdomain grid for error computation
  - `--save_img_paper` for drawing figures used in paper
  - `--plot_n` subdomain grid for drawing figures used in paper
