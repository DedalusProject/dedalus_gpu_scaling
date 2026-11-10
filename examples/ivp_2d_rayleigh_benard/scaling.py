"""Scaling script for 2D Rayleigh-Benard on GPU."""

import numpy as np
import rayleigh_benard as sim


resolutions = [(64, 16), (128, 32), (256, 64), (512, 128), (1024, 256)]
speeds = []
dtype = np.float64

for Nx, Nz in resolutions:
    timestep = 0.5 / Nz
    iterations = 1000
    outputs = False
    stats = False
    stats = sim.run(Nx, Nz, timestep, iterations, dtype, outputs=outputs, stats=stats)
    speeds.append(stats['speed'])

np.savez("scaling_data.npz", resolutions=resolutions, speeds=speeds, dtype=dtype)
