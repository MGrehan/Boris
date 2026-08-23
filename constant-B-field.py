#%%
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
#%%

## functions can be made much faster using  
# @jit(nopython=True)
# above them (push, E and B) but be warned this will cache the function 
# and you will need to restart kernal to change the funcion!

# Particle class definition
class Particle:
    def __init__(self, mass, charge):
        self.mass = mass
        self.charge = charge
        self.r = np.zeros(3, dtype='float64')
        self.u = np.zeros(3, dtype='float64')
        self.gamma = 1.0  # Set initial gamma

    def initPos(self, x, y, z):
        self.r[:] = (x, y, z)

    def initSpeed(self, ux, uy, uz):
        self.u[:] = (ux, uy, uz)


# Push function for particle simulation
def push(r, u, gamma, charge, mass, dt):
    rplus = r + u * dt / (2 * gamma)
    u += charge * E(rplus) * dt / (2 * mass)
    gamma_minus = np.sqrt(1 + np.sum(u ** 2))

    # Calculate magnetic field effect
    B_effect = charge * dt * B(rplus) / (2 * mass * gamma_minus)
    t = B_effect
    u1 = u + np.cross(u, t)
    s  = 2*t/(1 + np.sum(t*t))
    u += np.cross(u1, s)

    u += charge * E(rplus) * dt / (2 * mass)
    gamma = np.sqrt(1 + np.sum(u ** 2))

    r = rplus + dt * u / (2 * gamma)
    return r, u, gamma

def simulate_particle(particle_params):
    charge, mass, r0, u0, dt, Nt = particle_params
 
    part = Particle(mass, charge)
    part.initPos(r0[0], r0[1], r0[2])
    part.initSpeed(u0[0], u0[1], u0[2])
 
    # Prepare arrays to hold trajectory data
    x = np.zeros(Nt)
    y = np.zeros(Nt)
    z = np.zeros(Nt)
    ux = np.zeros(Nt)
    uy = np.zeros(Nt)
    uz = np.zeros(Nt)
    gamma = np.zeros(Nt)
    t = np.zeros(Nt)
    time = 0
   
    for i in range(Nt):
        part.r, part.u, part.gamma = push(part.r, part.u, part.gamma, charge, mass, dt)
        x[i], y[i], z[i] = part.r
        ux[i], uy[i], uz[i] = part.u
        gamma[i] = part.gamma
        t[i] = time
        time = time + dt
 
    return x, y, z, ux, uy, uz, gamma, t

#%%
################################################################################
############################    CONSTANT B FIELD    ############################
################################################################################



# Electric field function
def E(r, E0=0.1):
    # return np.array([E0 * np.cos(np.pi * r[1])/ 2, E0 * np.cos(np.pi * r[2])/ 2, 0.0])
    return np.array([0.0, 0.0, 0.0])

# Magnetic field function
def B(r, B0=1):
    # return np.array([B0 * np.sin(np.p i * r[2])/ 2, B0 * np.sin(np.pi * r[1])/ 2, B0])
    return np.array([0, 0, B0])

charge, mass = 1.0, 1.0

# Random initial positions and speeds for particles
r0 = np.array([0,0,0])
u0 = np.array([0.1,0.1,0.01])

# Calculate initial parameters
gamma0 = np.sqrt(1 + np.sum(u0 ** 2))
w0 = np.abs(charge) * np.sqrt(np.sum(B(r0) * B(r0))) / (mass / (gamma0))
dt = 0.0001 / w0
Np = 10.0  # Number of periods 
Tf = Np * 2 * np.pi / w0
Nt = int(Tf // dt)

particle_params_list = [charge, mass, r0, u0, dt, Nt]



x, y, z, ux, uy, uz, gamma, t= simulate_particle(particle_params_list)

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, color='r')

ax.set_xlabel('$x$', labelpad=15)
ax.set_ylabel('$y$', labelpad=15)
ax.set_zlabel('$z$', labelpad=5)
plt.show()
plt.close(fig)
# %%

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(t*w0/(2*np.pi), gamma)
ax.set_xlabel('$t/(2\\pi/\\omega)$')
ax.set_ylabel('$\\Gamma$')
plt.show()
plt.close()

# %%

# first build segments: a list of (point_i, point_{i+1})
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# create a norm and choose a cmap
norm = plt.Normalize((t*w0/(2*np.pi)).min(), (t*w0/(2*np.pi)).max())
cmap = plt.cm.viridis

# make the Line3DCollection
lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=2)
# set the values used for colormapping
lc.set_array(t*w0/(2*np.pi))
lc.set_linewidth(2)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection(lc)

# set limits (important so autoscaling sees your data)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())

# labels
ax.set_xlabel('$x$', labelpad=15)
ax.set_ylabel('$y$', labelpad=15)
ax.set_zlabel('$z$', labelpad=5)

# colorbar above
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])   # only needed for the colorbar
cbar = fig.colorbar(
    mappable,
    ax=ax,
    orientation='horizontal',
    pad=0.0,      # space between plot and colorbar
    aspect=50,
    fraction = 0.05,
    location = 'top',
    shrink=0.5,     # <-- make it 50% as long as the axes
)
cbar.set_label('$t/(2\\pi/\\omega)$')

plt.show()
plt.close(fig)
# %%

