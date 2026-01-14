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

    r += dt * u / gamma
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


################################################################################
############################    CONSTANT E FIELD    ############################
################################################################################

## Uniform E, no B
# Electric field function
E0 = 0.1
def E(r, E0=0.1):
    return np.array([0, 0, E0])

# Magnetic field function
def B(r, B0=1):
    return np.array([0, 0, 0])

charge, mass = 1.0, 1.0

# initial positions and 4 velocity for particles
r0 = np.array([0,0,0])
u0 = np.array([0.0,0.0,0.0])

# Calculate initial parameters
gamma0 = np.sqrt(1 + np.sum(u0 ** 2))
dt = 0.01
tf = 20
Nt = int(tf/dt)

particle_params_list = [charge, mass, r0, u0, dt, Nt]



x, y, z, ux, uy, uz, gamma, t= simulate_particle(particle_params_list)
# %%

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(t, gamma)
ax.plot(t, np.sqrt(1 + (charge*E0*t/mass)**2), 'k--')
ax.set_xlabel('$t$')
ax.set_ylabel('$\\Gamma$')
ax.set_title('Constant $\\bm{E}=E_0 \\hat{z}$')
ax.set_xlim(0,t.max())
plt.show()
plt.close()

# %%
