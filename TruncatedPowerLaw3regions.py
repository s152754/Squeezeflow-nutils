from nutils import cli, mesh, function, solver, export, types
import treelog, numpy, typing, pandas

unit = types.unit(m=1, s=1, g=1e-3, K=1, N='kg*m/s2', Pa='N/m2', J='N*m', W='J/s', bar='0.1MPa', min='60s', hr='60min')


def main(mu0: unit['Pa*s'], muinf: unit['Pa*s'], tcr: unit['s'], nu: float, R0: unit['m'], H0: unit['m'], F: unit['N'], T: unit['s'], m: int,
         degree: int, npicard: int, tol: float, nt: int, ratio: int, plotnewtonian: bool):
    '''
    Radial squeeze flow of a non-newtonian fluid
    .. arguments::
       mu0 [1.Pa*s]
         Viscosity at zero shear rate
       muinf [0.001Pa*s]
         Viscosity at infinite shear rate
       tcr [0.890s]
         Newton/Power law cross over time scale
       nu [0.5]
         Power index
       R0 [2cm]
         Initial radius of fluid domain
       H0 [1mm]
         Initial height of fluid domain
       F [5.N]
         Loading
       T [300ms]
         Final time
       m [16]
         Number of (radial) elements
       ratio [1000]
         Ratio between largest and smallest time step
       nt [100]
         Number of time steps
       npicard [50]
         Number of Picard iterations
       tol [0.001]
         Tolerance for Picard iterations
       degree [3]
         Pressure order in radial direction
       plotnewtonian [True]
         Flag to plot the Newtonian reference solution
    .. presets::
       newtonian
         nu=1.
         mu0=1Pa*s
         plotnewtonian=True
       carreau
         nu=0.05
         mu0=200Pa*s
         F=5.0N
         R0=1.0cm
         h0=0.5mm
         plotnewtonian=True
         T=5s
 '''

    assert 0. < nu <= 1.0

    # initialize namespace with constants
    ns = function.Namespace()
    ns.π = numpy.pi

    # physical parameters
    ns.F    = F
    ns.μ0   = mu0
    ns.μinf = muinf
    ns.tcr  = tcr
    ns.ν    = nu
    ns.K    = mu0 * tcr ** (nu - 1.)

    # domain and geometry definition
    domain, ρgeom = mesh.line(numpy.linspace(0, 1, m + 1), bnames=('inner', 'outer'), space='R')
    h0 = H0 / 2
    ns.V = numpy.pi * H0 * R0 ** 2
    ns.h = '?h'
    ns.h0 = '?h0'
    ns.hpicard = '?hpicard'
    ns.R = 'sqrt(V / (2 π h))'
    ns.R0 = 'sqrt(V / (2 π h0))'
    ns.Rpicard = 'sqrt(V / (2 π hpicard))'
    ns.ρ = ρgeom
    ns.r = 'ρ Rpicard'
    ns.x = function.stack([ns.r])
    ns.δh = '(h - h0) / ?Δt'
    ns.w1picard = 'hpicard'  # using mid point works well
    ns.w2picard = 'hpicard'  # probably should start from top

    # pressure discretization
    ns.basis = domain.basis('spline', degree)
    ns.p = 'basis_i ?lhs_i'
    ns.ppicard = 'basis_i ?lhspicard_i'

    ns.w1picard = function.min(ns.hpicard, abs('-((K)^(1 / (1 - ν))) (μ0^(ν / (ν - 1))) / ppicard_,0' @ ns))
    # ns.w2picard = function.min(ns.hpicard, abs('-(K)^(1 / (1 - ν)) μinf^(ν / (ν - 1)) / ppicard_,0' @ ns))
    ns.w2picard = function.min(ns.hpicard, '(μinf^(ν / (ν - 1)) / μ0^(ν / (ν - 1))) w1picard' @ ns)
    # ns.w1picard = ns.hpicard / 1

    # flux definition
    # ns.Cnewton      = '-((2 r) / (3 μ0)) (hpicard)^3'
    # ns.Cpower       = 0
    # ns.Cnewtonhigh  = 0

    # ns.Cnewton = '2 r ( (-(w1picard^3) / (3 μ0)) + ((ν / (1 + ν)) (1 / K)^(1 / ν) (abs(ppicard_,0))^(1 / ν - 1) ) (hpicard^(1 / ν + 1) w1picard - w1picard^(1 / ν + 2)) )'
    # ns.Cpower  = '2 r ( (ν / (1 + ν)) (1 / K)^(1 / ν) (abs(ppicard_,0))^((1 / ν) - 1) ((((1 + ν) / (1 + 2 ν)) hpicard^((1 / ν) + 2)) - (hpicard^((1 / ν) + 1) w1picard) + ((ν / (1 + 2 ν)) w1picard^((1 / ν) + 2))))'
    # ns.Cnewtonhigh = 0

    ns.Cnewton      = '2 r ( -(w1picard^3 / (3 μ0)) + (1 / (1 + 1 / ν)) (1 / K)^(1 / ν) (abs(ppicard_,0))^(1 / ν - 1) ( w2picard^(1 / ν + 1) w1picard - w1picard^(1 / ν + 2) ) + (1 / (2 μinf)) ( w2picard^2 w1picard - hpicard^2 w1picard ) )'
    ns.Cpower       = '2 r ( (1 / (1 + 1 / ν)) (1 / K)^(1 / ν) (abs(ppicard_,0))^(1 / ν - 1) ( ((1 + ν) / (1 + 2 ν)) w2picard^(1 / ν + 2) - w2picard^(1 / ν + 1) w1picard + (1 / (1 / ν + 2)) w1picard^(1 / ν + 2) ) )'
    ns.Cnewtonhigh  = '2 r ( (1 / (2 μinf)) ( (-2 / 3) hpicard^3 - (1 / 3) w2picard^3 + hpicard^3 w2picard ) )'

    ns.Q = '(Cnewton + Cpower + Cnewtonhigh) p_,0'

    # pressure boundary condition
    sqr = domain.boundary['outer'].integral('p^2' @ ns, degree=4)
    cons = solver.optimize('lhs', sqr, droptol=1e-10)

    # definition of the residuals
    resp = domain.integral('(2 δh basis_i r - Q basis_i,0) d:x' @ ns, degree=4)
    resh = domain.integral('((F / (Rpicard^2)) - π p) r d:x' @ ns, degree=4)

    # initialization
    state = {'h0': h0, 'lhs0': numpy.zeros_like(cons)}
    pp = PostProcessing(domain, ns(**state), plotnewtonian)

    # time iteration
    Δtsteps = numpy.power(ratio ** (1 / (nt - 1)), range(nt))
    Δtsteps = (T / Δtsteps.sum()) * Δtsteps
    treelog.user(f'Min Δt: {numpy.min(Δtsteps):4.2e}s')
    treelog.user(f'Max Δt: {numpy.max(Δtsteps):4.2e}s')
    with treelog.iter.plain('timestep', range(nt)) as steps:
        for step in steps:

            state['Δt'] = Δtsteps[step]
            state['t'] = Δtsteps[:step + 1].sum()
            state['hpicard'] = state['h0']
            state['lhspicard'] = state['lhs0']

            with treelog.iter.plain('picard', range(npicard)) as iterations:
                for iteration in iterations:

                    # solve the linear system of equations
                    state = solver.solve_linear(['lhs', 'h'], [resp, resh], constrain={'lhs': cons}, arguments=state)

                    # check convergence
                    herror = abs(state['hpicard'] - state['h']) / abs(state['h0'])
                    treelog.user(f'height error = {herror:4.3e}')
                    if herror < tol:
                        break

                    relax = 1.0
                    state['lhspicard'] = (1 - relax) * state['lhspicard'] + relax * state['lhs']
                    state['hpicard'] = (1 - relax) * state['hpicard'] + relax * state['h']

            # postprocessing
            pp.plot(state)

            # set initial conditions for the next time step
            state['lhs0'] = state['lhs']
            state['h0'] = state['h']

    # return the time series data frame
    return pp.df


class PostProcessing:

    def __init__(self, domain, ns, plotnewtonian, npoints=6):
        self.interior = domain.sample('bezier', npoints)
        self.ns = ns
        self.df = pandas.DataFrame({'t': [0.], 'h': [ns.h0.eval()], 'R': [ns.R0.eval()]})
        self.plotana = plotnewtonian

    def plot(self, state):

        ns = self.ns(**state)  # copy namespace so that we don't modify the calling argument

        # plots
        x, p, h, w1, w2, dp = self.interior.eval(['r', 'p', 'h', 'w1picard', 'w2picard', 'p_,0'] @ ns)

        with export.mplfigure('pressure.png') as fig:
            ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='p [Pa]')
            ax.plot(x / unit('mm'), p / unit('Pa'))
            ax.grid()
        with export.mplfigure('pressure gradient.png') as fig:
            ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='dp/dr [Pa/mm]')
            ax.plot(x / unit('mm'), dp / (unit('Pa')/unit('mm')))
            ax.grid()

        with export.mplfigure('height.png') as fig:
            ax = fig.add_subplot(111, xlabel='r [mm]', ylabel='h [mm]', ylim=(0, 1.1 * numpy.max(h)))
            ax.plot(x / unit('mm'), h, label='$h$')
            ax.plot(x / unit('mm'), w1, label='$w1$')
            ax.plot(x / unit('mm'), w2, label='$w2$')
            ax.grid()
            ax.legend()

        # plot the constitutive relation
        K   = ns.K.eval()
        nu  = ns.ν.eval()
        mu0 = ns.μ0.eval()
        muinf = ns.μinf.eval()

        with export.mplfigure('viscosity.png') as fig:
            ax = fig.add_subplot(111, xlabel='$\dot{\gamma}$ [1/s]', ylabel='$\mu$ [Pa·s]')
            for hr, w1r, dpr in zip(h, w1, dp):
                treelog.user(hr,w1r)
                zr = numpy.linspace(0, hr, 10)
                γ1 = - 1 / mu0 * dpr * zr
                γ2 = numpy.power(- (1 / K) * dpr * zr, 1 / nu)
                γ3 = - 1 / muinf * dpr * zr
                mu2 = K * γ2 ** (nu - 1)
                mu1 = mu0 * numpy.ones_like(mu2)
                mu3 = muinf * numpy.ones_like(mu2)
                i = (mu2 < mu1).astype(int) + (mu2 < mu3).astype(int)
                mu = numpy.choose(i, [mu1, mu2, mu3])
                γ = numpy.choose(i, [γ1, γ2, γ3])
                ax.loglog(γ / unit('1/s'), mu / unit('Pa*s'),'.')
                # ax.loglog(γ2 / unit('1/s'), mu2 / unit('Pa*s'),'.')
                # ax.loglog(γ3 / unit('1/s'), mu3 / unit('Pa*s'), '.')
                # γdot = (1 / ns.tcr.eval()) * numpy.power(10, numpy.linspace(-3, 3, 100))
            # mu = numpy.minimum(mu0 * numpy.ones_like(γdot), ns.K.eval() * γdot ** (nu - 1))
            ax.grid()

        # time plots
        self.df = self.df.append({'t': state['t'], 'h': state['h'], 'R': ns.R.eval()}, ignore_index=True)

        t_ana = numpy.linspace(0, self.df['t'].max(), 10)
        h0 = self.df['h'][0]
        R0 = self.df['R'][0]

        F = ns.F.eval()
        μ0 = ns.μ0.eval()
        h_ana = h0 * (1 + (8 * F * (2 * h0) ** 2) / (3 * numpy.pi * μ0 * (R0 ** 4)) * t_ana) ** (-1 / 4)
        R_ana = R0 * numpy.sqrt(h0 / h_ana)

        with export.mplfigure('radius.png') as fig:
            ax = fig.add_subplot(111, xlabel='t [s]', ylabel='R [mm]')
            ax.plot(self.df['t'] / unit('s'), self.df['R'] / unit('mm'), '.-', label='FE')
            if self.plotana:
                ax.plot(t_ana / unit('s'), R_ana / unit('mm'), ':', label='analytical (Newtonian)')
            ax.grid()
            ax.legend()

        with export.mplfigure('semiheight.png') as fig:
            ax = fig.add_subplot(111, xlabel='$t$ [s]', ylabel='$h$ [mm]')
            ax.plot(self.df['t'] / unit('s'), self.df['h'] / unit('mm'), '.-', label='$h$')
            if self.plotana:
                ax.plot(t_ana / unit('s'), h_ana / unit('mm'), ':', label='analytical (Newtonian)')
            ax.grid()
            ax.legend()

        # save data frame to file
        with treelog.userfile('timeseries.csv', 'w') as f:
            self.df.to_csv(f, index=False)


cli.run(main)