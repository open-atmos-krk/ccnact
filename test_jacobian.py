from ccnact import parcel
import timeit
import time

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
    n_iters = 2
    times = [0, 0, 0]

    # act

    for i in range(3):
        exec_time = 0
        for _ in range(n_iters):
            t1 = time.time()
            parcel(**example_parameters, analytic_jacobian=i)
            exec_time += time.time() - t1
        times[i] = exec_time / n_iters

    # assert
    print()
    print("none  vs analytic", times[0] / times[2])
    print("scipy vs analytic", times[1] / times[2])
    print("scipy vs none    ", times[1] / times[0])
    assert False
    assert times[0] / times[2] > 17,  "Analytic jacobian not 17 times faster no jacobian"
