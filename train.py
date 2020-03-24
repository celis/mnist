from mnist.service.data_service import DataService
from mnist.model.logistic_regression import LogisticRegression
from mnist.model.trainer import Trainer
from mnist.datasource.datasource import CustomDataource
from mnist.configuration import Configuration
from mnist.service.s3 import S3
import logging


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    logging.info("loading configuration")
    config = Configuration("configs/model_config.json").parameters

    logging.info("loading MNIST data")
    data_service = DataService()
    data_service.save(config["data_path"])

    datasource = CustomDataource(config["data_path"])
    train_dataset, validation_dataset = datasource.transform()

    model = LogisticRegression()

    logging.info("training model")
    trainer = Trainer(model=model, **config["trainer"])
    trainer.fit(train_dataset, validation_dataset)

    logging.info("storing model artifacts on disk")
    trainer.save(config["model_artifact"])

    logging.info("uploading model artifact to S3")
    aws_config = Configuration("configs/aws_config.json").parameters["s3"]
    s3 = S3(**aws_config)
    filename, key = config["model_artifact"], config["model_artifact"].split("/")[1]
    s3.upload(filename, key)
