import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize((256,256))])

batch_size = 36

trainset = torchvision.datasets.ImageFolder(root='../Train', transform=transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
testset = torchvision.datasets.ImageFolder(root='../Test', transform=transform)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, 3)
        self.conv2 = nn.Conv2d(7, 13, 5)
        self.fc1 = nn.Linear(48373, 84)
        self.fc2 = nn.Linear(84, 8)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
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
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    acc = 100*correct
    return acc, test_loss
    print(f"Test Error: \n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 15
train_acc = []
train_loss = []
test_acc = []
test_loss = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    tr_acc, tr_loss = train(train_dataloader, model, loss_fn, optimizer)
    train_acc.append(tr_acc)
    train_loss.append(tr_loss)
    te_acc, te_loss = test(test_dataloader, model, loss_fn)
    test_acc.append(te_acc)
    test_loss.append(te_loss)
print("Done!")

# plot loss
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.plot(range(len(test_loss)), test_loss, label="test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

# plot training accuracy
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.plot(range(len(test_acc)), test_acc, label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy (%)")
plt.legend()
plt.grid()
plt.show()