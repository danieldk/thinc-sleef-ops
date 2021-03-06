from typing import Callable, Union
import math
import numpy as np
import pytest

from thinc_sleef_ops import InstructionSet, SleefOps, with_cpu_feature

M_SQRT1_2 = 1.0 / math.sqrt(2.0)
M_1_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

numpy_erf = np.vectorize(math.erf)


def numpy_logistic_cdf(x):
    return 1.0 / (1.0 + np.exp(-x))


def numpy_logistic_pdf(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def numpy_cdf(x):
    return 0.5 * (1.0 + numpy_erf(x * M_SQRT1_2))


def numpy_pdf(x):
    return M_1_SQRT_2PI * np.exp(-0.5 * x**2)


def numpy_softmax(x):
    maxes = np.max(x, axis=-1, keepdims=True)
    shifted = x - maxes
    new_x = np.exp(shifted)
    new_x /= new_x.sum(axis=-1, keepdims=True)
    return new_x


def test_inputs():
    return [
        # Trigger at least one back-off to a more narrow instruction set.
        np.random.normal(size=(61,)) * 10,
        np.random.normal(size=(10, 2)) * 10,
        np.random.normal(size=(5, 9, 3)) * 10,
    ]


@pytest.fixture
def ops():
    return SleefOps()


def check_elementwise_function(
    op_name: str,
    f_check: Callable[[np.ndarray], np.ndarray],
    cpu_feature: InstructionSet,
    dtype: Union[np.float32, np.float64],
    inplace: bool,
    inputs: np.ndarray,
):
    with with_cpu_feature(cpu_feature) as feature_ops:
        f = getattr(feature_ops, op_name)
        inputs_copy = inputs.copy()

        assert np.allclose(f(inputs_copy, inplace=inplace), f_check(inputs), atol=1e-4)
        if inplace:
            assert np.allclose(inputs_copy, f_check(inputs), atol=1e-4)


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("X", test_inputs())
def test_exp(ops, cpu_feature, dtype, inplace, X):
    check_elementwise_function("exp", np.exp, cpu_feature, dtype, inplace, X)


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_erff_torch(ops, cpu_feature, dtype, inplace):
    ERF_CHECK = np.array(
        [
            -1.0000,
            -0.9996,
            -0.9953,
            -0.9661,
            -0.8427,
            -0.5205,
            0.0000,
            0.5205,
            0.8427,
            0.9661,
            0.9953,
            0.9996,
            1.0000,
        ],
        dtype=dtype,
    )

    inputs = np.arange(-3, 3.5, 0.5, dtype=dtype)
    check_elementwise_function(
        "erf", lambda _: ERF_CHECK, cpu_feature, dtype, inplace, inputs
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("X", test_inputs())
def test_gelu(ops, cpu_feature, dtype, inplace, X):
    check_elementwise_function(
        "gelu",
        lambda x: x * numpy_cdf(x),
        cpu_feature,
        dtype,
        inplace,
        X,
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("X", test_inputs())
def test_gelu_backward(ops, cpu_feature, dtype, inplace, X):
    check_elementwise_function(
        "gelu_backward",
        lambda x: numpy_cdf(x) + x * numpy_pdf(x),
        cpu_feature,
        dtype,
        inplace,
        X,
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_gelu_backward_torch(ops, cpu_feature, dtype, inplace):
    GELU_GRADS_CHECK = [
        -0.0119,
        -0.0376,
        -0.0852,
        -0.1275,
        -0.0833,
        0.1325,
        0.5000,
        0.8675,
        1.0833,
        1.1275,
        1.0852,
        1.0376,
        1.0119,
    ]
    inputs = np.arange(-3, 3.5, 0.5, dtype=dtype)
    check_elementwise_function(
        "gelu_backward", lambda _: GELU_GRADS_CHECK, cpu_feature, dtype, inplace, inputs
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("X", test_inputs())
def test_sigmoid(ops, cpu_feature, dtype, inplace, X):
    check_elementwise_function(
        "sigmoid",
        numpy_logistic_cdf,
        cpu_feature,
        dtype,
        inplace,
        X,
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("X", test_inputs())
def test_softmax(ops, cpu_feature, dtype, inplace, X):
    check_elementwise_function(
        "softmax",
        numpy_softmax,
        cpu_feature,
        dtype,
        inplace,
        X,
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("X", test_inputs())
def test_swish(ops, cpu_feature, dtype, inplace, X):
    check_elementwise_function(
        "swish",
        lambda x: x * numpy_logistic_cdf(x),
        cpu_feature,
        dtype,
        inplace,
        X,
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("X", test_inputs())
def test_swish_backward(ops, cpu_feature, dtype, inplace, X):
    check_elementwise_function(
        "swish_backward",
        lambda x: numpy_logistic_cdf(x) + x * numpy_logistic_pdf(x),
        cpu_feature,
        dtype,
        inplace,
        X,
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
def test_swish_backward_torch(ops, cpu_feature, dtype, inplace):
    SWISH_GRADS_CHECK = [
        -0.0881,
        -0.0994,
        -0.0908,
        -0.0413,
        0.0723,
        0.2600,
        0.5000,
        0.7400,
        0.9277,
        1.0413,
        1.0908,
        1.0994,
        1.0881,
    ]
    inputs = np.arange(-3, 3.5, 0.5, dtype=dtype)
    check_elementwise_function(
        "swish_backward",
        lambda _: SWISH_GRADS_CHECK,
        cpu_feature,
        dtype,
        inplace,
        inputs,
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("X", test_inputs())
def test_tanh(ops, cpu_feature, dtype, inplace, X):
    check_elementwise_function("tanh", np.tanh, cpu_feature, dtype, inplace, X)
