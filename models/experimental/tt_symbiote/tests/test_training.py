# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for training with TTNN backend."""

import torch
from torch import nn

from models.experimental.tt_symbiote.modules.linear import TTNNLinearTraining
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


class RandomLinear(nn.Module):
    """A simple linear model for testing purposes."""

    def __init__(self, input_dim, output_dim):
        super(RandomLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x * 2.0
        output = self.linear(x)
        return output


# First define the LinearFunction (from previous example)
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


# Define ModuleLinear
class ModuleLinear:
    def forward(self, input, weight, bias=None):
        return LinearFunction.apply(input, weight, bias)


# Usage example
class MyModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Create parameters manually
        self.linear = nn.Linear(in_features, out_features)
        self.linear_grad = ModuleLinear()

    def forward(self, x):
        return self.linear.forward(x)


def test_training_with_ttnn(device):
    """Test training a simple model with TTNN acceleration."""
    # Model configuration
    input_dim = 10
    output_dim = 300
    batch_size = 4
    num_epochs = 50
    learning_rate = 0.001

    # Create model
    model = MyModel(input_dim, output_dim)

    # Replace PyTorch modules with TTNN equivalents
    nn_to_ttnn = {
        nn.Linear: TTNNLinearTraining,
    }

    # Set model to training mode
    model.train()
    torch.set_grad_enabled(True)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)

    # Generate synthetic training data
    X_train = TorchTTNNTensor(torch.randn(100, input_dim))
    # Create targets with some relationship to inputs for learning
    y_train = TorchTTNNTensor(torch.randn(100, output_dim))

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = len(X_train) // batch_size

        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)

            # Compute loss
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("\nTraining complete!")

    # Test inference
    model.eval()
    torch.set_grad_enabled(False)
    test_input = torch.randn(2, input_dim)
    test_output = model(test_input)
    print(f"\nTest inference output shape: {test_output.shape}")
    print(f"Test output:\n{test_output}")
