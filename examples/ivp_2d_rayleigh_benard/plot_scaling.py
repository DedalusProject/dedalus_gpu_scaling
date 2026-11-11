
import numpy as np
import matplotlib.pyplot as plt

data = np.load("scaling_data.npz", allow_pickle=True)

plt.figure(figsize=(5, 5))
plt.semilogx(np.prod(data['resolutions'], axis=1), data['speed'], 'o-')
plt.xlabel('N = Nx*Nz')
plt.ylabel('Speed (mode-stages/cpu-sec)')
plt.ylim(0, None)
plt.grid()
plt.title(data['title'])
plt.tight_layout()
plt.savefig("speed_scaling.pdf")
