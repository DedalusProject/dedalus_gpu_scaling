"""Scaling script for 2D Rayleigh-Benard."""

import numpy as np
from mpi4py import MPI
import rayleigh_benard as sim
try:
    import cupy as cp
    array_namespace = cp
except ImportError:
    array_namespace = np


# Scaling run parameters
Nz = [16, 24, 32, 48, 64, 96, 128, 192, 256]
resolutions = [(4*n, n) for n in Nz]
dtype = np.float64

# Gather statistics
stats = []
for Nx, Nz in resolutions:
    timestep = 0.5 / Nz
    iterations = 100
    stats.append(sim.run(Nx, Nz, timestep, iterations, dtype, array_namespace, outputs=False, stats=False))


if MPI.COMM_WORLD.rank == 0:

    # Coalesce statistics
    data = {}
    for key in stats[0].keys():
        data[key] = np.array([s[key] for s in stats])

    # Save data
    data['resolutions'] = resolutions
    data['dtype'] = dtype
    data['array_namespace'] = array_namespace.__name__
    if array_namespace.__name__ == 'cupy':
        data['title'] = "2D RBC on GPU"
    else:
        data['title'] = f"2D RBC on CPU ({MPI.COMM_WORLD.Get_size()} cores)"
    np.savez("scaling_data.npz", **data)
