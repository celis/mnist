import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch import optim
import torch
import logging


class Trainer:
    """
    Trainer class with methods taking care of model training and storage of model artifact
    """

    def __init__(self, model, batch_size: int, epochs: int, lr: float):
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.loss_function = F.cross_entropy
        self.optimizer = optim.SGD(model.parameters(), lr=lr)

    def _get_data(
        self, train_dataset: TensorDataset, validation_dataset: TensorDataset
    ):
        """
        Generates DataLoader instances for training and validation data
        """
        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            DataLoader(validation_dataset, batch_size=self.batch_size * 2),
        )

    def _loss_batch(self, x: torch.Tensor, y: torch.Tensor, optimizer=None):
        """
        Updates model parameters with a forward and backward pass
        on a batch, returns loss value and batch size
        """
        loss = self.loss_function(self.model(x), y)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return loss.item(), len(x)

    def fit(self, train_dataset: TensorDataset, valid_dataset: TensorDataset):
        """
        Performs model training
        """

        (train_dataloader, valid_dataloader) = self._get_data(
            train_dataset, valid_dataset
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for x, y in train_dataloader:
                self._loss_batch(x, y, self.optimizer)

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[self._loss_batch(xb, yb) for xb, yb in valid_dataloader]
                )
            loss_value = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            logging.info(f" epoch: { epoch }, loss: { loss_value }")

    def save(self, path: str):
        """
        Stores model artifact in 'path'
        """
        torch.save(self.model.state_dict(), path)
