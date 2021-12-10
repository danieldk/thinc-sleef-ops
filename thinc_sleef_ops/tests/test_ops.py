import numpy as np
import pytest

import thinc_sleef_ops


@pytest.fixture
def ops():
    return thinc_sleef_ops.SleefOps()


def test_exp(ops):
    a = np.arange(-3, 3.5, 0.5, dtype=np.float32)
    assert np.allclose(ops.exp(a), np.exp(a))


def test_erff(ops):
    a = np.arange(-3, 3.5, 0.5, dtype=np.float32)
    assert np.allclose(
        ops.erf(a),
        np.array(
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
            ]
        ),
        atol=1e-4,
    )


def test_tanh(ops):
    a = np.arange(-10, 10, 0.5, dtype=np.float32)
    assert np.allclose(ops.tanh(a), np.tanh(a))
