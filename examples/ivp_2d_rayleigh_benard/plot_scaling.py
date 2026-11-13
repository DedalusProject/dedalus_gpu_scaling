
import numpy as np
import matplotlib.pyplot as plt

data = np.load("scaling_data.npz", allow_pickle=True)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].loglog(np.prod(data['resolutions'], axis=1), data['speed'], 'o-')
ax[0].set_xlabel('N = Nx*Nz')
ax[0].set_ylabel('Speed (mode-stages/cpu-sec)')
ax[0].grid()
ax[1].semilogx(np.prod(data['resolutions'], axis=1), data['speed'], 'o-')
ax[1].set_xlabel('N = Nx*Nz')
ax[1].set_ylabel('Speed (mode-stages/cpu-sec)')
ax[1].set_ylim(0, None)
ax[1].grid()
plt.title(data['title'])
plt.tight_layout()
plt.savefig("speed_scaling.pdf")
