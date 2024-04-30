import torch
from agent.networks import CNN
import torch


class BCAgent:

    def __init__(self, n_classes=3, history_length=0, lr=0.001):
        # TODO: Define network, loss function, optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")
        self.net = CNN(history_length=history_length, n_classes=n_classes).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        self.history_length = history_length

    def update(self, X_batch, y_batch):
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        if self.history_length == 0: X_batch = X_batch.unsqueeze(1)
        prediction = self.net(X_batch)
        loss = self.loss_fn(prediction, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        X = X.unsqueeze(0)
        outputs = self.net(X)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
        return file_name
