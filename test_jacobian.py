from ccnact import parcel


def test_jacobian(example_parameters):
    # arrange
    output = {}

    # act
    for analytic_jacobian in [False]:
        output[analytic_jacobian] = parcel(
            **example_parameters, analytic_jacobian=analytic_jacobian
        )

    # assert
    # TODO: check with Jacobian makes it faster
    # TODO: check results equivalent (up to epsilon...)
