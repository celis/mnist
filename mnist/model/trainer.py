import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.optim import SGD
import torch
import logging


class Trainer:
    """
    """

    def __init__(self, model, epochs: int, lr: float, batch_size: int):
        self.model = model
        self.optimizer = SGD(model.parameters(), lr=lr)
        self.loss_function = F.cross_entropy
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

    def _get_data(
        self, train_dataset: TensorDataset, validation_dataset: TensorDataset
    ):
        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            DataLoader(validation_dataset, batch_size=self.batch_size * 2),
        )

    def _loss_batch(self, x, y):
        loss = self.loss_function(self.model(x), y)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item(), len(x)

    def fit(self, train_dataset, validation_dataset):
        train_dataloader, validation_dataloader = self._get_data(
            train_dataset, validation_dataset
        )

        for epoch in range(self.epochs):
            self.model.train()
            for x, y in train_dataloader:
                self._loss_batch(self.loss_function, x, y, self.optimizer)

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[
                        self.loss_batch(self.loss_function, x, y)
                        for x, y in validation_dataloader
                    ]
                )
            loss_value = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            logging.info(f"epoch: { epoch }, loss: { loss_value }")
