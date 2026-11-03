import pytest


@pytest.fixture(name="example_parameters")
def _example_parameters():
    return dict(
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
    )
