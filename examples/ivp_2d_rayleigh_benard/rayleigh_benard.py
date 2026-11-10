"""2D Rayleigh-Benard on GPU."""

import numpy as np
import cupy as cp
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


def run(Nx, Nz, timestep, iterations, dtype, outputs=False, stats=False):

    # Fixed parameteres
    Lx, Lz = 4, 1
    Rayleigh = 2e6
    Prandtl = 1
    dealias = 3/2
    timestepper = d3.RK222

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype, array_namespace=cp)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
    problem.add_equation("b(z=0) = Lz")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_iteration = iterations

    # Initial conditions
    b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    b['g'] *= z * (Lz - z) # Damp noise at walls
    b['g'] += Lz - z # Add linear background

    # Analysis
    if outputs:
        snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
        snapshots.add_task(b, name='buoyancy')
        snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    # Flow properties
    if stats:
        flow = d3.GlobalFlowProperty(solver, cadence=10)
        flow.add_property(np.sqrt(u@u)/nu, name='Re')

    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                if stats:
                    max_Re = flow.max('Re')
                    logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
                else:
                    logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        return solver.log_stats()


if __name__ == "__main__":

    # Parameters
    Nx = 256
    Nz = 64
    timestep = 1e-3
    iterations = 1000
    dtype = np.float64
    outputs = False
    stats = True

    run(Nx, Nz, timestep, iterations, dtype, outputs=outputs, stats=stats)
