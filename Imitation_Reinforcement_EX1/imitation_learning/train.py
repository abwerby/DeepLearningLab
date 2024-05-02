import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torch
import time 

import sys

sys.path.append(".")

import utils
from agent.bc_agent import BCAgent
from torch.utils.tensorboard import SummaryWriter


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("[INFO] read data")
    data_file = os.path.join(datasets_dir, "data.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):
    """
    Preprocess the states and actions.
    :param X_train: training states
    :param y_train: training actions
    :param X_valid: validation states
    :param y_valid: validation actions
    :param history_length: number of previous states to concatenate to the current state
    :return: preprocessed data (X_train, y_train, X_valid, y_valid)
    """
    
    print("[INFO] preprocess data")

    
    # convert to gray scale
    X_train = utils.rgb2gray(X_train)
    X_valid = utils.rgb2gray(X_valid)
    
    # standardize images
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_valid = (X_valid - X_valid.mean()) / X_valid.std()
    
    # normalize images between 0 and 1
    # X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    # X_valid = (X_valid - X_valid.min()) / (X_valid.max() - X_valid.min())
    
    # clean labels
    y_train = utils.clean_labels(y_train)
    y_valid = utils.clean_labels(y_valid)
    
    # convert labels
    y_train = utils.convert_labels(y_train)
    y_valid = utils.convert_labels(y_valid)
    
    # upsampling the data, so that the classes are balanced
    X_train, y_train = utils.upsample(X_train, y_train)
    X_valid, y_valid = utils.upsample(X_valid, y_valid)
    # Add History
    if history_length > 0:
        X_train, y_train= utils.add_history(X_train, y_train, history_length)
        X_valid, y_valid = utils.add_history(X_valid, y_valid, history_length)
    # convert images to tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)

    return X_train, y_train, X_valid, y_valid


class BCDataset(torch.utils.data.Dataset):
    """
    Dataset class for imitation learning.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(
    X_train,
    y_train,
    X_valid,
    batch_size,
    lr,
    history_length=1,
    epochs=10,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("[INFO] train model")

    # define agent
    agent = BCAgent(n_classes=5, history_length=history_length, lr=lr)
    time_ = time.strftime("%Y-%m-%d_%H-%M-%S")
    expermints_name = f"IM-time_{time_}"
    writer = SummaryWriter(tensorboard_dir + "/" + expermints_name)
    # write hyperparameters to tensorboard
    writer.add_hparams({"lr": lr, "batch_size": batch_size, "history_length": history_length}, {})
    # create data loader for training and validation data
    train_dataset = BCDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    print(f'len(train_loader): {len(train_loader)}')
    vaild_dataset = BCDataset(X_valid, y_valid)
    valid_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=batch_size, shuffle=False)
    print(f'len(valid_loader): {len(valid_loader)}')
    
    # training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        acu_train_loss = 0
        acu_val_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            loss = agent.update(X_batch, y_batch)
            acu_train_loss += loss
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + i)
        print(f"Loss: {acu_train_loss/len(train_loader)}")
        writer.add_scalar('Loss/train', acu_train_loss/len(train_loader), epoch)
        # validate
        for i, (X_batch, y_batch) in enumerate(valid_loader):
            loss = agent.update(X_batch, y_batch)
            acu_val_loss += loss
            writer.add_scalar('Loss/val_batch', loss.item(), epoch * len(valid_loader) + i)
        writer.add_scalar('Loss/val', acu_val_loss/len(valid_loader), epoch)
        print(f"Val Loss: {acu_val_loss/len(valid_loader)}")
        
    # save agent weigths
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)
    # close tensorboard writer
    writer.flush()
    writer.close()


if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")
    # preprocess data
    history_length = 7
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=history_length
    )
    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, batch_size=64, lr=1e-4, history_length=history_length, epochs=100)
