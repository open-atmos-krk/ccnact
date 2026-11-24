from ccnact import parcel
import time
import numpy as np

def test_analytical_jacobian_against_scipy(example_parameters):
    # TODO: check results equivalent (up to epsilon...)
    # arrange
    output = {}

    # act


    # assert


def test_jacobian_execution_time(example_parameters):
    # TODO: check with Jacobian makes it faster
    # TODo check dependence on n_iters
    # arrange
    n_iters = 50
    n_warmup = 5
    times = {0: [], 2: []}

    for i in [0, 2]:
        for _ in range(n_warmup):
            parcel(**example_parameters, analytic_jacobian=i)

    # act
    for i in [0, 2]:
        for _ in range(n_iters):
            start = time.perf_counter()
            parcel(**example_parameters, analytic_jacobian=i)

            times[i].append(time.perf_counter() - start)

    speedup = np.median(times[0]) / np.median(times[2])

    expected_speedup_macos = 5.2
    print(speedup) # FIXME it is for checking different platforms
    assert speedup > expected_speedup_macos
