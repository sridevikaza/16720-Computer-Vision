import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# load data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# convert to Tensor
train_x = torch.from_numpy(train_x).to(torch.float32)
train_y = np.argmax(train_y, axis=1)
train_y = torch.from_numpy(train_y)
train_data = TensorDataset(train_x,train_y)
# convert to Tensor
test_x = torch.from_numpy(test_x).to(torch.float32)
test_y = np.argmax(test_y, axis=1)
test_y = torch.from_numpy(test_y)
test_data = TensorDataset(train_x,train_y)
# convert to Tensor
valid_x = torch.from_numpy(valid_x).to(torch.float32)
valid_y = np.argmax(valid_y, axis=1)
valid_y = torch.from_numpy(valid_y)
valid_data = TensorDataset(valid_x,valid_y)

batch_size = 32

# Create data loaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 7, 3)
        self.conv2 = nn.Conv2d(7, 13, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(325, 84)  # 5*5 from image dimension
        self.fc2 = nn.Linear(84, 36)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.view(-1, 1, 32, 32)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss = train_loss + loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    train_loss /= num_batches
    print(f"Train Accuracy: {(100 * correct):>0.1f}%")
    return 100*correct, train_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(-1, 1, 32, 32)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    acc = 100*correct
    return acc, test_loss
    print(f"Test Error: \n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(-1, 1, 32, 32)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    acc = 100*correct
    return acc, test_loss
    print(f"Validation Error: \n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 15
train_acc = []
train_loss = []
validate_acc = []
validate_loss = []
test_acc = []
test_loss = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    tr_acc, tr_loss = train(train_dataloader, model, loss_fn, optimizer)
    train_acc.append(tr_acc)
    train_loss.append(tr_loss)
    val_acc, val_loss = validate(valid_dataloader, model, loss_fn)
    validate_acc.append(val_acc)
    validate_loss.append(val_loss)
    te_acc, te_loss = test(test_dataloader, model, loss_fn)
    test_acc.append(te_acc)
    test_loss.append(te_loss)
print("Done!")

# plot loss
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.plot(range(len(validate_loss)), validate_loss, label="validation")
plt.plot(range(len(test_loss)), test_loss, label="test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

# plot training accuracy
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.plot(range(len(validate_acc)), validate_acc, label="validation")
plt.plot(range(len(test_acc)), test_acc, label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy (%)")
plt.legend()
plt.grid()
plt.show()