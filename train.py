from mnist.service.data_service import DataService
from mnist.model.logistic_regression import LogisticRegression
from mnist.model.trainer import Trainer
from mnist.datasource.datasource import CustomDataource
from mnist.configuration import Configuration
import logging


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    config = Configuration("config/model_config.json").parameters

    data_service = DataService()
    data_service.save(config["data_path"])

    datasource = CustomDataource(config["data_path"])
    train_dataset, validation_dataset = datasource.transform()

    model = LogisticRegression()

    trainer = Trainer(model =model, **config["trainer"])
    trainer.fit(train_dataset, validation_dataset )
    trainer.save(config["model_artifact"])





