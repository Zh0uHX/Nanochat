import pytest
import torch

from nanochat.optim import (
    _adamw_step_compiled,
    _adamw_step_eager,
    _muon_step_compiled,
    _muon_step_eager,
)


def adamw_inputs(device="cpu"):
    generator = torch.Generator(device=device).manual_seed(7)
    parameter = torch.randn(8, 8, generator=generator, device=device)
    gradient = torch.randn(8, 8, generator=generator, device=device)
    first_moment = torch.zeros_like(parameter)
    second_moment = torch.zeros_like(parameter)
    scalars = [
        torch.tensor(value, dtype=torch.float32)
        for value in (3.0, 1e-3, 0.9, 0.95, 1e-8, 0.01)
    ]
    return parameter, gradient, first_moment, second_moment, *scalars


def test_eager_adamw_matches_explicit_reference():
    inputs = adamw_inputs()
    parameter, gradient, first_moment, second_moment, *scalars = inputs
    expected_parameter = parameter.clone()
    expected_first = first_moment.clone()
    expected_second = second_moment.clone()
    step, learning_rate, beta1, beta2, epsilon, weight_decay = [
        value.item() for value in scalars
    ]
    expected_parameter.mul_(1 - learning_rate * weight_decay)
    expected_first.lerp_(gradient, 1 - beta1)
    expected_second.lerp_(gradient.square(), 1 - beta2)
    bias1 = 1 - beta1**step
    bias2 = 1 - beta2**step
    denominator = (expected_second / bias2).sqrt() + epsilon
    expected_parameter.add_(
        expected_first / denominator,
        alpha=-(learning_rate / bias1),
    )

    _adamw_step_eager(*inputs)

    torch.testing.assert_close(parameter, expected_parameter)
    torch.testing.assert_close(first_moment, expected_first)
    torch.testing.assert_close(second_moment, expected_second)


@pytest.mark.slow
def test_compiled_adamw_matches_eager():
    eager_inputs = adamw_inputs()
    compiled_inputs = tuple(
        value.clone() if isinstance(value, torch.Tensor) else value
        for value in eager_inputs
    )

    _adamw_step_eager(*eager_inputs)
    _adamw_step_compiled(*compiled_inputs)

    for eager_value, compiled_value in zip(eager_inputs[:4], compiled_inputs[:4]):
        torch.testing.assert_close(eager_value, compiled_value)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Muon parity requires CUDA")
def test_compiled_muon_matches_eager_on_cuda():
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(11)
    gradients = torch.randn(
        2, 16, 32, generator=generator, device=device, dtype=torch.bfloat16
    )
    parameters = torch.randn(
        2, 16, 32, generator=generator, device=device, dtype=torch.bfloat16
    )
    momentum = torch.zeros_like(gradients)
    second_momentum = torch.zeros(
        2, 16, 1, device=device, dtype=torch.bfloat16
    )
    scalars = [
        torch.tensor(value, dtype=torch.float32)
        for value in (0.9, 0.02, 0.01, 0.95)
    ]
    eager_inputs = (
        gradients.clone(),
        parameters.clone(),
        momentum.clone(),
        second_momentum.clone(),
        *[value.clone() for value in scalars],
        5,
        -1,
    )
    compiled_inputs = tuple(
        value.clone() if isinstance(value, torch.Tensor) else value
        for value in eager_inputs
    )

    _muon_step_eager(*eager_inputs)
    _muon_step_compiled(*compiled_inputs)

    torch.testing.assert_close(
        eager_inputs[1], compiled_inputs[1], atol=2e-2, rtol=2e-2
    )
