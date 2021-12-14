import numpy as np
import pytest

from thinc_sleef_ops import SleefOps, with_cpu_feature


@pytest.fixture
def ops():
    return SleefOps()


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exp(ops, cpu_feature, dtype):
    a = np.arange(-3, 3.5, 0.5, dtype=dtype)

    with with_cpu_feature(cpu_feature) as feature_ops:
        assert np.allclose(feature_ops.exp(a), np.exp(a))

    assert np.allclose(ops.exp(a), np.exp(a))


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_erff(ops, cpu_feature, dtype):
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

    a = np.arange(-3, 3.5, 0.5, dtype=dtype)

    with with_cpu_feature(cpu_feature) as feature_ops:
        assert np.allclose(feature_ops.erf(a), ERF_CHECK, atol=1e-4)

    assert np.allclose(
        ops.erf(a),
        ERF_CHECK,
        atol=1e-4,
    )


@pytest.mark.parametrize("cpu_feature", SleefOps.instruction_sets())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tanh(ops, cpu_feature, dtype):
    a = np.arange(-10, 10, 0.5, dtype=dtype)

    with with_cpu_feature(cpu_feature) as feature_ops:
        assert np.allclose(feature_ops.tanh(a), np.tanh(a), atol=1e-4)

    assert np.allclose(ops.tanh(a), np.tanh(a))
