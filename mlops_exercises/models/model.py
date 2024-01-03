import torch
import torch.nn as nn


class MyNeuralNet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.l1 = torch.nn.Linear(in_features, 500)
        self.l2 = torch.nn.Linear(500, out_features)
        self.r = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.l2(self.r(self.l1(x)))


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, 10), nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
