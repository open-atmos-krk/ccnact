from ccnact import parcel
import pytest

@pytest.mark.parametrize(
    "use_analytic_jacobian",
    [False, True],
)
def test_jacobian(example_parameters, use_analytic_jacobian):
    # arrange
    output = {}

    # act
    output[use_analytic_jacobian] = parcel(
        **example_parameters, analytic_jacobian=use_analytic_jacobian
    )

    # assert
    # TODO: check with Jacobian makes it faster
    # TODO: check results equivalent (up to epsilon...)
