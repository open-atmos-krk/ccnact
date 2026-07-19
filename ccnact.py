"""
A minimal, yet complete, adiabatic parcel model with moving-sectional/particle-based
aerosol/cloud microphysics representation capturing condensation, CCN [de]activation,
ripening and evaporation.

Code based on the "Exploring Cloud Microphysics Modeling Concepts the Pythonic Way"
paper draft. Dependencies are: NumPy, SciPy, pytest and Pint (which is de-facto only
used within tests, by default all computations are done on pure floats).
"""

# pylint: disable=non-ascii-name,fixme,no-member,import-error,global-statement
# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments

import math
from collections import namedtuple
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable

import numpy as np
import pint
import scipy
from scipy.optimize import elementwise as scipy_optimize_elementwise

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass


def parcel(
    *,
    sigma: float = 0.072,
    kappa: Iterable[float],
    meanr: Iterable[float],
    n_tot: Iterable[float],
    gstdv: Iterable[float],
    MAC: float = 1,
    n_bins: float,
    RH: float,
    w: float,
    T: float,
    p: float,
    nt: int,
    dt: float = None,
    R_d: float = None,
    R_v: float = None,
    l_v: float = None,
    g: float = None,
    c_pd: float = None,
    rho_l: float = None,
    D_v: float = None,
    stop_at_s_max: bool = False,
    method: str = "LSODA",
    rtol: float = 1e-4,
    allow_ripening: bool = True,
):
    """runs a simulation for the given parameters and returns a tuple of:
    - concentration of droplets with r_wet>r_crit at final timestep (in metre^{-3} @ STP);
    - relative humidity (saturation) profile during the ascent (1D float array);
    - wet radii (2D float array);
    - time coordinate (1D float array);
    - activation status defined by r_w>r_c condition (2D Boolea array).
    """
    assert len(kappa) == len(meanr) == len(n_tot) == len(gstdv)
    assert all(np.asarray(kappa) == kappa[0])  # TODO
    assert MAC == 1  # TODO

    c, si = constants(R_d=R_d, R_v=R_v, l_v=l_v, g=g, c_pd=c_pd, rho_l=rho_l, D_v=D_v)
    _, ix, s = cfg_ccn(
        c,
        si,
        w=w,
        sigma=sigma,
        RH=RH,
        T=T,
        n_sd=n_bins,
        kappa=kappa,
        meanr=meanr,
        n_tot=n_tot,
        gstdv=gstdv,
        nt=nt,
        dt=dt or 1 * si.s,
        p=p,
    )
    y0 = initial_condition(eqp, c, s, ix)
    sol = solve(
        c=c,
        s=s,
        ix=ix,
        y0=y0,
        stop_at_s_max=stop_at_s_max,
        method=method,
        rtol=rtol,
        allow_ripening=allow_ripening,
    )

    p_d = sol.y[ix.p_d] * si.Pa
    T = sol.y[ix.T] * si.K
    r_w = eqp.r_w(c, x=sol.y[ix.x])
    RH = eqp.RH(
        q_v=s.q_t - eqp.q_l(c, s, r_w=r_w),
        ρ_vs=eqp.ρ_v(c, p_v=eqp.p_vs(c, T=T), T=T),
        ρ_d=eqp.ρ_d(c, p_d=p_d, T=T),
    )
    act = r_w > eqp.r_c(c, s, r_d=s.r_d[:, None], T=T[None, :])
    n_a = act[:, -1] @ s.ξ / s.m_d * c.ρ_stp
    time = sol.t * si.s
    return n_a, RH, r_w, time, act


def cfg_ccn(
    c,
    si,
    *,
    w=2.5,
    sigma=0.072,
    RH=0.91,
    T=300,
    p=106216,
    n_sd=44,
    meanr=(3.2e-8,),
    gstdv=(1.75,),
    n_tot=(8e9,),
    kappa=(0.7,),
    nt=150,
    dt=1,
):
    """returns a tuple of (internally used): config dict, indices namedtuple
    and setup namedtuple; the default function arguments match the setup
    used in the GMD draft"""
    v_m3_stp = 1
    cfg = {
        "dist": tuple(
            scipy.stats.lognorm(  # dry CCN spectrum
                s := math.log(gstdv[m]),
                scale=(meanr[m]) / math.exp((s**2) / 2),
            )
            for m in range(len(n_tot))
        ),
        "norm": tuple(n_tot[m] * v_m3_stp for m in range(len(n_tot))),
        "Δv_m3_stp": v_m3_stp,
        "n_sd": n_sd,
        "κ": kappa[0],
    }
    cfg["r_d"], cfg["ξ"] = quantile_sample(
        dist=cfg["dist"],
        n_sd=cfg["n_sd"],
        norm=cfg["norm"],
    )
    e = RH * eqp.p_vs(c, T * si.K)
    q_t = c.eps * e / (p * si.Pa - e)
    return (
        cfg,
        namedtuple(
            "Indices",
            (
                ix := {
                    "x": slice(0, cfg["n_sd"]),
                    "p_d": cfg["n_sd"],
                    "T": cfg["n_sd"] + 1,
                    "size": cfg["n_sd"] + 2,
                }
            ),
        )(**ix),
        namedtuple(
            "Setup",
            (
                s := {
                    "p_d0": p * si.Pa - e,  # initial dry pres.
                    "t_max": nt * dt * si.s,  # temporal extent
                    "q_t": q_t,  # total water
                    "w": w * si.m / si.s,  # updraft speed
                    "T_0": T * si.K,  # initial temp.
                    "κ": cfg["κ"],  # hygroscopicity
                    "σ_w": sigma * si.J / si.m**2,  # surf tension coef.
                    "m_d": cfg["Δv_m3_stp"] * si.m**3 * c.ρ_stp,
                    "r_d": cfg["r_d"] * si.m,
                    "ξ": cfg["ξ"],
                }
            ),
        )(**s),
    )


def quantile_sample(*, dist, n_sd, norm):
    """samples muti-mode lognormal size distribution on `n_sd` size sections
    with a uniform-in-multiplicity-per-mode layout, in which each mode is
    represented with `n_sd // len(dist)` super-particles"""
    n_modes = len(dist)
    n_per_mode = n_sd // n_modes
    assert n_per_mode * n_modes == n_sd
    q = (np.arange(n_per_mode) + 0.5) / n_per_mode
    return (
        np.concatenate(tuple(dist[m].ppf(q) for m in range(n_modes))),
        np.concatenate(
            tuple(
                np.full(n_per_mode, norm[m] / n_per_mode, dtype=np.int64)
                for m in range(n_modes)
            )
        ),
    )


def constants(*, R_d=None, R_v=None, l_v=None, g=None, c_pd=None, rho_l=None, D_v=None):
    """returns a constants catalogue instantiated as a `nametuple` with default values
    optionally replaced with values from the arguments; if run under the
    `DimensionalAnalysis` context manager, all values are instantiated as Pint quantities
    """
    si = SI

    zero_c = scipy.constants.zero_Celsius * si.K
    p_stp = 101.325 * si.kPa  # ICAO

    R_d = R_d or 287.0 * si.J / si.K / si.kg
    R_v = R_v or 461.5 * si.J / si.K / si.kg
    l_v = l_v or 2.5e6 * si.J / si.kg
    g = g or scipy.constants.g * si.m / si.s**2
    c_pd = c_pd or 1 * si.kJ / si.kg / si.K
    rho_l = 1 * si.kg / si.liter
    D_v = 0.25 * si.cm**2 / si.s

    c = namedtuple(
        "Consts",
        (
            tmp := {
                "R_d": R_d,
                "R_v": R_v,
                "eps": R_d / R_v,
                "T_0C": zero_c,
                "c_pd": c_pd,
                "l_v": l_v,
                "g": g,
                "ρ_w": rho_l,
                "D_v": D_v,
                "ρ_stp": p_stp / R_d / (15 * si.K + zero_c),  # ICAO
                "r_0": 1 * si.nm,  # for ln(r/r_0)
                # Bolton 1980 saturation vapor pressure coeffs
                # (https://doi.org/10.1175/1520-0493(1980)108%3C1046:TCOEPT%3E2.0.CO;2)
                "B80_G0": 6.112 * si.hPa,
                "B80_G1": 17.67,
                "B80_G2": 243.5 * si.K,
            }
        ),
    )(**tmp)
    return c, si


def __ode_helper(y, e, c, s, ix, allow_ripening):
    """common calculations for ODE solution"""
    ρ_d = e.ρ_d(c, p_d=y[ix.p_d], T=y[ix.T])
    ρ_vs = e.ρ_v(c, p_v=e.p_vs(c, T=y[ix.T]), T=y[ix.T])
    r_w = e.r_w(c, x=y[ix.x])
    dr_w__dt = e.dr_w__dt(
        c,
        r_w=r_w,
        ρ_v=ρ_vs * (rh := e.RH(ρ_vs=ρ_vs, ρ_d=ρ_d, q_v=s.q_t - e.q_l(c, s, r_w=r_w))),
        ρ_o=ρ_vs * e.RH_eq(c, s, r_w=r_w, r_d=s.r_d, T=y[ix.T]),
    )
    if rh > 1 and not allow_ripening:
        dr_w__dt = np.maximum(dr_w__dt, 0)
    dq_v__dt = e.dq_v__dt(c, s, r_w=r_w, dr_w__dt=dr_w__dt)
    return ρ_d, ρ_vs, r_w, dr_w__dt, dq_v__dt


eqp = namedtuple(
    "Eqs",
    (
        eqp := {
            "RH_eq": lambda c, s, r_w, r_d, T: (r_w**3 - r_d**3)
            / (r_w**3 - r_d**3 * (1 - s.κ))
            * np.exp(2 * s.σ_w / (c.R_v * T * c.ρ_w * r_w)),
            "dp_d__dt": lambda c, p, ρ_d: -p.w * ρ_d * c.g,
            "dT__dt": lambda c, dp_d__dt, dq_v__dt, ρ_d: (
                dp_d__dt / ρ_d - dq_v__dt * c.l_v
            )
            / c.c_pd,
            "dr_w__dt": lambda c, r_w, ρ_v, ρ_o: c.D_v / c.ρ_w * (ρ_v - ρ_o) / r_w,
            "x": lambda c, r_w: np.log(r_w / c.r_0),
            "r_w": lambda c, x: c.r_0 * np.exp(x),
            "dr_w__dx": lambda r_w: r_w,
            "ρ_d": lambda c, p_d, T: p_d / c.R_d / T,
            "ρ_v": lambda c, p_v, T: p_v / c.R_v / T,
            "p_vs": lambda c, T: c.B80_G0
            * np.exp((c.B80_G1 * (T - c.T_0C)) / ((T - c.T_0C) + c.B80_G2)),
            "RH": lambda q_v, ρ_vs, ρ_d: ρ_d * q_v / ρ_vs,
            "q_l": lambda c, s, r_w: 4 / 3 * np.pi * (c.ρ_w * s.ξ @ r_w**3) / s.m_d,
            "dq_v__dt": lambda c, s, r_w, dr_w__dt: -4
            * np.pi
            * (s.ξ * r_w**2 @ dr_w__dt)
            / s.m_d
            * c.ρ_w,
            "r_c": lambda c, s, r_d, T: (
                3 * s.κ * r_d**3 / (2 * s.σ_w / (c.R_v * T * c.ρ_w))
            )
            ** 0.5,
            "ode_helper": __ode_helper,
        }
    ).keys(),
)(**eqp)


def ode_rhs(_, y, dy__dt, e, c, s, ix, allow_ripening=True):
    """ODE system right-hand-side following eq. (13) in Arabas & Shima 2017
    (https://doi.org/10.5194/npg-24-535-2017)"""
    ρ_d, _, r_w, dr_w__dt, dq_v__dt = e.ode_helper(y, e, c, s, ix, allow_ripening)
    dy__dt[ix.p_d] = e.dp_d__dt(c, s, ρ_d=ρ_d)
    dy__dt[ix.x] = dr_w__dt / e.dr_w__dx(r_w=r_w)
    dy__dt[ix.T] = e.dT__dt(c, dp_d__dt=dy__dt[ix.p_d], dq_v__dt=dq_v__dt, ρ_d=ρ_d)
    return dy__dt


def stop_cond(_, y, __, e, c, s, ix):
    """returns the time derivative of RH, following eq. 7.22 in Rogers & Yau book;
    the value is used to detect maximal saturation stopping point"""
    ρ_d, ρ_vs, __, ___, dq_v__dt = e.ode_helper(y, e, c, s, ix)
    q1 = (c.eps * c.l_v * c.g / c.R_d / c.c_pd / y[ix.T] - c.g / c.R_d) / y[ix.T]
    q2 = ρ_d * (
        1 / ρ_vs + c.eps * c.l_v**2 / y[ix.p_d] / y[ix.T] / c.c_pd
    )  # TODO: rho vs. rho_d; p vs. pd
    return q1 * s.w + q2 * dq_v__dt


stop_cond.terminal = True


def initial_condition(e, c, s, ix):
    """returns a state vector with wet radii in equilibrium with the dry radii"""
    cmn = {"c": c, "T": s.T_0}
    rh = e.RH(
        q_v=s.q_t,
        ρ_d=e.ρ_d(p_d=s.p_d0, **cmn),
        ρ_vs=e.ρ_v(p_v=e.p_vs(**cmn), **cmn),
    )
    cmn |= {"s": s}
    root = scipy_optimize_elementwise.find_root(
        lambda x, r_d: rh - e.RH_eq(r_w=x, r_d=r_d, **cmn),
        (s.r_d, e.r_c(r_d=s.r_d, **cmn)),
        args=(s.r_d,),
    )
    assert all(root.success)

    y0 = np.empty(ix.size)
    y0[ix.p_d] = s.p_d0
    y0[ix.T] = s.T_0
    y0[ix.x] = e.x(c, r_w=root.x)
    return y0


def solve(*, c, s, ix, y0, stop_at_s_max, method, rtol, allow_ripening):
    """performs time integration using SciPy's interface to LSODA"""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sol = scipy.integrate.solve_ivp(
            ode_rhs,
            (0, s.t_max),
            y0,
            args=(np.empty(ix.size), eqp, c, s, ix, allow_ripening),
            method=method,
            rtol=rtol,
            events=stop_cond if stop_at_s_max else None,
        )
    assert sol.success, sol.message
    return sol


def _fake(si_unit):
    return (1.0 * si_unit).to_base_units().magnitude


class FakeUnitRegistry:  # pylint: disable=too-few-public-methods
    """a mock of pint.UnitRegistry() with a subset of the API, but working on pure floats"""

    def __init__(self, si):
        self.dimensionless = 1.0
        for prefix in ("nano", "micro", "milli", "centi", "deci", "", "hecto", "kilo"):
            for unit in (
                "metre",
                "meter",
                "gram",
                "mole",
                "joule",
                "kelvin",
                "second",
                "minute",
                "pascal",
                "litre",
                "liter",
                "hour",
                "newton",
            ):
                self.__setattr__(prefix + unit, _fake(si.__getattr__(prefix + unit)))
                self.__setattr__(
                    prefix + unit + "s", _fake(si.__getattr__(prefix + unit + "s"))
                )

        for prefix in ("n", "u", "m", "c", "d", "", "h", "k"):
            for unit in (
                "m",
                "g",
                "mol",
                "J",
                "K",
                "s",
                "min",
                "day",
                "Pa",
                "l",
                "h",
                "bar",
                "N",
            ):
                self.__setattr__(prefix + unit, _fake(si.__getattr__(prefix + unit)))


PINT_SI = pint.UnitRegistry()
SI = FakeUnitRegistry(PINT_SI)

# TESTS ########################################################################

if "pytest" in str(__loader__):

    import platform
    from contextlib import nullcontext
    from pathlib import Path

    import pytest
    from open_atmos_jupyter_utils import notebook_vars

    class DimensionalAnalysis:
        """context manager enabling true Pint unit checks"""

        def __enter__(*_):  # pylint: disable=no-method-argument,no-self-argument
            global SI
            SI = PINT_SI

        def __exit__(*_):  # pylint: disable=no-method-argument,no-self-argument
            global SI
            SI = FakeUnitRegistry(PINT_SI)

    @pytest.mark.parametrize(
        "ctx",
        (
            pytest.param(nullcontext(), id="fake units"),
            pytest.param(DimensionalAnalysis(), id="real units"),
        ),
    )
    def test_constants(ctx):
        """calls the constants() function with and without the unit handling"""
        with ctx:
            _ = constants()

    def test_ode_rhs():
        """checks unit correctness in the ODE definition"""
        with DimensionalAnalysis():
            # Arrange
            c, si = constants()
            _, ix, s = cfg_ccn(c, si)
            y = [np.nan] * ix.size
            y[ix.T] *= si.K
            y[ix.p_d] *= si.Pa

            # Act
            rhs = ode_rhs(None, y, [np.nan] * ix.size, eqp, c, s, ix)

            # Assert
            assert rhs[ix.T].check("[temperature] / [time]")
            assert rhs[ix.p_d].check("[pressure] / [time]")
            assert all(x.check("1 / [time]") for x in rhs[ix.x])

    def test_stop_cond():
        """checks unit correctness in the stopping condition definition"""
        with DimensionalAnalysis():
            # Arrange
            c, si = constants()
            _, ix, s = cfg_ccn(c, si)
            y = [np.nan] * ix.size
            y[ix.T] *= si.K
            y[ix.p_d] *= si.Pa

            # Act
            drh_dt = stop_cond(None, y, [np.nan] * ix.size, eqp, c, s, ix)

            # Assert
            assert drh_dt.check("1 / [time]")

    def test_dimensional_analysis():
        """checks if the dimensional analysis logic throws an error on bogus addition"""
        with pytest.raises(Exception) as excinfo:
            with DimensionalAnalysis():
                c, _ = constants()
                __ = c.T_0C + c.g
        assert "Cannot convert from 'kelvin' ([temperature]) to" in str(excinfo.value)

    def test_case_from_the_paper():
        """repropoduces simulation from the GMD paper draft asserting on the final values"""
        c, si = constants()
        _, ix, s = cfg_ccn(c, si)
        y0 = initial_condition(eqp, c, s, ix)
        sol = solve(
            c=c,
            s=s,
            ix=ix,
            y0=y0,
            stop_at_s_max=False,
            method="LSODA",
            rtol=1e-4,
            allow_ripening=True,
        )

        with DimensionalAnalysis():
            c, si = constants()
            _, ix, s = cfg_ccn(c, si)

            p_d = sol.y[ix.p_d] * si.Pa
            temp = sol.y[ix.T] * si.K
            r_w = eqp.r_w(c, x=sol.y[ix.x])
            rh = eqp.RH(
                q_v=s.q_t - eqp.q_l(c, s, r_w=r_w),
                ρ_vs=eqp.ρ_v(c, p_v=eqp.p_vs(c, T=temp), T=temp),
                ρ_d=eqp.ρ_d(c, p_d=p_d, T=temp),
            )
            r_c = eqp.r_c(c, s, r_d=s.r_d[:, None], T=temp[None, :])

            n_a = (r_w[:, -1] > r_c[:, -1]) @ s.ξ / s.m_d * c.ρ_stp
            err = np.amax(s.ξ) / s.m_d * c.ρ_stp

        # assert
        assert f"{min(rh):.2g~}" == "0.91"
        assert f"{max(rh - 1):.2g~}" == "0.0016"
        assert f"{n_a.to(u := si.cm**-3):.4g~}" == "1091 / cm ** 3"
        assert f"{err.to(u):.3g~}" == "182 / cm ** 3"

    @pytest.mark.parametrize("stop_at_s_max", (True, False))
    def test_parcel(stop_at_s_max):
        """runs the parcel() interface with arbitrary parameters asserting on the
        returned values"""
        n1_act, rh, _, _, _ = parcel(
            w=1,
            kappa=(0.8, 0.8),
            meanr=(3e-8, 3e-8),
            n_tot=(0.5e9, 0.5e9),
            gstdv=(1.5, 1.5),
            n_bins=100,
            RH=0.99,
            T=300,
            p=1e5,
            MAC=1,
            sigma=0.072,
            dt=2,
            nt=100,
            R_d=287.0558,
            R_v=461.5,
            l_v=2500712,
            g=9.80665,
            c_pd=1004.6,
            rho_l=1,
            D_v=2.26e-05,
            stop_at_s_max=stop_at_s_max,
        )
        np.testing.assert_approx_equal(max(rh), 1.002026)
        np.testing.assert_approx_equal(n1_act, 220e6)

    class TestExampleBasics:
        """testing results of computation in the notebook (incl. time measurements)"""

        # pylint:disable=missing-function-docstring

        @staticmethod
        @pytest.fixture(scope="session", name="nb_vars_basics")
        def variables_fixture_basics():
            return notebook_vars(
                file=Path(__file__).parent / "examples" / "basics.ipynb", plot=False
            )

        @staticmethod
        @pytest.fixture(scope="session", name="nb_vars_raw_state")
        def variables_fixture_raw_state():
            return notebook_vars(
                file=Path(__file__).parent / "examples" / "raw_state.ipynb", plot=False
            )

        @staticmethod
        def test_raw_state_nb(nb_vars_raw_state):
            pass  # TODO

        @staticmethod
        @pytest.mark.parametrize("var", ("s_max", "n_act"))
        def test_monotonic_vars(nb_vars_basics, var):
            assert all(np.diff(nb_vars_basics[var])) > 0

        @staticmethod
        @pytest.mark.parametrize(
            "var, index, value",
            (
                ("s_max", 0, 1.00117),
                ("n_act", 0, 80e6),
                ("s_max", -1, 1.00421),
                ("n_act", -1, 78.5e7),
            ),
        )
        def test_check_values(nb_vars_basics, var, index, value):
            np.testing.assert_approx_equal(
                nb_vars_basics[var][index], value, significant=5
            )

        @staticmethod
        @pytest.mark.xfail(
            platform.system() == "Darwin" and platform.machine() == "x86_64",
            reason="GitHub workers performance seem to vary",
        )
        def test_no_regression_in_walltime(nb_vars_basics):
            assert nb_vars_basics["wall_time"] < 15 * SI.s

        @staticmethod
        def test_concurrent(nb_vars_basics):
            assert (
                nb_vars_basics["cpu_time"]
                >= {
                    "Linux": 2.5,
                    "Darwin": 0.75,
                    "Windows": 1,
                }[platform.system()]
                * nb_vars_basics["wall_time"]
            )
