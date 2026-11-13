"""Scaling script for 2D Rayleigh-Benard."""

import numpy as np
from mpi4py import MPI
from collections import defaultdict
import rayleigh_benard as sim
import logging
logger = logging.getLogger(__name__)
try:
    import cupy as cp
    array_namespace = cp
except ImportError:
    array_namespace = np


# Scaling run parameters
Nz = [16, 24, 32, 48, 64, 96, 128, 192, 256, 512]
resolutions = [(4*n, n) for n in Nz]
dtype = np.float64

# Gather statistics
data = defaultdict(list)
data['resolutions'] = resolutions
data['dtype'] = dtype
data['array_namespace'] = array_namespace.__name__
if array_namespace.__name__ == 'cupy':
    data['title'] = "2D RBC on GPU"
else:
    data['title'] = f"2D RBC on CPU ({MPI.COMM_WORLD.Get_size()} cores)"
for Nx, Nz in resolutions:
    # Run
    logger.info(f"Running simulation with Nx={Nx}, Nz={Nz}")
    timestep = 0.5 / Nz
    iterations = 100
    stats = sim.run(Nx, Nz, timestep, iterations, dtype, array_namespace, outputs=False, stats=False)
    # Save stats
    if MPI.COMM_WORLD.rank == 0:
        for key, value in stats.items():
            data[key].append(value)
        np.savez("scaling_data.npz", **data)
