import numpy as np
import pytest

import thinc_sleef_ops.cy as cy


@pytest.fixture
def ops():
    return cy.SleefArray()


@pytest.mark.parametrize("cpu_feature", cy.SleefArray.instruction_sets())
def test_exp(ops, cpu_feature):
    a = np.arange(-3, 3.5, 0.5, dtype=np.float32)

    with cy.with_cpu_feature(cpu_feature) as feature_ops:
        assert np.allclose(feature_ops.exp(a), np.exp(a))

    assert np.allclose(ops.exp(a), np.exp(a))


@pytest.mark.parametrize("cpu_feature", cy.SleefArray.instruction_sets())
def test_erff(ops, cpu_feature):
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
        ]
    )

    a = np.arange(-3, 3.5, 0.5, dtype=np.float32)

    with cy.with_cpu_feature(cpu_feature) as feature_ops:
        assert np.allclose(feature_ops.erf(a), ERF_CHECK, atol=1e-4)

    assert np.allclose(
        ops.erf(a),
        ERF_CHECK,
        atol=1e-4,
    )


@pytest.mark.parametrize("cpu_feature", cy.SleefArray.instruction_sets())
def test_tanh(ops, cpu_feature):
    a = np.arange(-10, 10, 0.5, dtype=np.float32)

    with cy.with_cpu_feature(cpu_feature) as feature_ops:
        assert np.allclose(feature_ops.tanh(a), np.tanh(a), atol=1e-4)

    assert np.allclose(ops.tanh(a), np.tanh(a))
