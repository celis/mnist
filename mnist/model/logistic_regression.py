from torch import nn


class LogisticRegression(nn.Module):
    """
    Implements Logistic Regression model with Pytorch
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
