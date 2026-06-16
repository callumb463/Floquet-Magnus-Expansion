import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



data = np.load('error_grids.npz')
decay_rate = data['decay_rate']
epsilons = data['epsilons']
omegas = data['omegas']

fig, ax = plt.subplots()

# decay_rate < 0: ||F_n|| shrinking with order (series converging)
# decay_rate > 0: ||F_n|| growing with order (series diverging)
norm = colors.TwoSlopeNorm(vmin=decay_rate.min(), vcenter=0.0, vmax=decay_rate.max())
im = ax.pcolormesh(epsilons, omegas, decay_rate, cmap='RdBu_r', norm=norm)
fig.colorbar(im, ax=ax, label=r'decay rate of $\Vert F_n \Vert$')
ax.contour(epsilons, omegas, decay_rate, levels=[0], colors='k', linewidths=1.5)
ax.set_xlabel(r'$\varepsilon$')
ax.set_ylabel(r'$\omega$')
ax.set_title('Magnus Series Convergence Boundary')

fig.tight_layout()
fig.savefig('plots/Phase_Diagram')
