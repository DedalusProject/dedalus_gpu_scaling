'''
Kolmogorov flow d3
'''
import numpy as np
import dedalus.public as d3
import logging
import cupy as cp

logger = logging.getLogger(__name__)

# Parameters
nu = 0.025
size = 1024
dtype = np.float32
timestepper = d3.RK222

# Bases
coords = d3.CartesianCoordinates('x', 'y')

dist   = d3.Distributor(coords, dtype=dtype, array_namespace=cp)
xbasis = d3.RealFourier(coords['x'], size=size, bounds=(0, 2*np.pi), dealias=3/2)
ybasis = d3.RealFourier(coords['y'], size=size, bounds=(0, 2*np.pi), dealias=3/2)

# Fields
psi = dist.Field(name='psi', bases=(xbasis, ybasis))
omega = dist.Field(name='omega', bases=(xbasis, ybasis))
f   = dist.Field(name='f', bases=ybasis)
f_u = dist.Field(name='f_u', bases=ybasis)
tau_psi = dist.Field(name='tau_psi')

# Substitutions
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
J  = lambda A, B: dx(A)*dy(B) - dy(A)*dx(B)

u = -dy(psi)
v =  dx(psi)

u_vec = d3.skew(d3.grad(psi)) 

# Forcing
x = dist.local_grid(xbasis)
y = dist.local_grid(ybasis)

f['g'] = - 4*np.cos(4*y)
f_u['g'] = np.sin(4*y)

# Initial condition
psi.fill_random(seed=6)
psi.low_pass_filter(scales=0.25)
psi['g'] *= 1e-8

# Problem
problem = d3.IVP([psi, omega, tau_psi], namespace=locals())
problem.add_equation("dt(omega) - nu*lap(omega) + tau_psi = -J(psi, omega) + f")
problem.add_equation("lap(psi) - omega = 0")
problem.add_equation("integ(psi) = 0")

# Build solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = 10

# Disabled for speed (refactorisations are expensive)
# CFL = d3.CFL(solver, initial_dt=1e-1, cadence=4, safety=0.8, threshold=0.1,
#              max_change=1.5, min_change=0.5, max_dt=1e-1)
# CFL.add_velocity(u_vec)

flow = d3.GlobalFlowProperty(solver, cadence=100)
flow.add_property(0.5*(u**2 + v**2)/(2*np.pi)**2, name='KE')
flow.add_property((u*f_u)/(2*np.pi)**2, name='I')
flow.add_property(nu*(dx(u)**2 + dy(u)**2 + dx(v)**2 + dy(v)**2)/(2*np.pi)**2, name='D')

timestep = 1e-3
# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        # timestep = CFL.compute_timestep()
        solver.step(timestep)
        if((solver.iteration-1) % 100 ==0 ):
            KE = flow.volume_integral('KE')
            I  = flow.volume_integral('I')
            D  = flow.volume_integral('D')
            logger.info('Iteration=%i, Time=%e, dt=%e, KE=%f, I=%f, D=%f' % (solver.iteration, solver.sim_time, timestep, KE, I, D))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
