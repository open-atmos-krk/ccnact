"""Tests for use with pytest"""

import os
import platform
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from open_atmos_jupyter_utils import notebook_vars

from ccnact import (
    SI,
    DimensionalAnalysis,
    cfg_ccn,
    constants,
    eqp,
    initial_condition,
    ode_rhs,
    parcel,
    solve,
    stop_cond,
)

# pylint: disable=non-ascii-name,fixme


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
    sol = solve(c=c, s=s, ix=ix, y0=y0, stop_at_s_max=False, method="LSODA", rtol=1e-4)

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
        np.testing.assert_approx_equal(nb_vars_basics[var][index], value, significant=5)

    @staticmethod
    @pytest.mark.xfail(
        platform.system() == "Darwin" and platform.machine() == "x86_64",
        reason="GitHub workers performance seem to vary",
    )
    @pytest.mark.xfail("CI" not in os.environ, reason="Checking only on CI")
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
