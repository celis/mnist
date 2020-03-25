# MNIST

Repository for training a classifier on the mnist datatset using PyTorch,
follows the tutorial [What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html).

# Usage

Create a virtual enviroment and install the required packages

    pip install -r requirements.txt
    
Adjust the config files and run

    python train.py
    
This will train a classifier based on Logistic Regression and upload the
model artifact to an AWS S3 bucket.   The model artifact will be used by
the web application [here](https://github.com/celis/flask_mnist).

