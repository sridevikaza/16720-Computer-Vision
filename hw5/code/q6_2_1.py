import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_1

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize((224,224))])
batch_size = 36

trainset = torchvision.datasets.ImageFolder(root='../data/oxford-flowers17/train', transform=transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
testset = torchvision.datasets.ImageFolder(root='../data/oxford-flowers17/test', transform=transform)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = squeezenet1_1(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

final_conv = nn.Conv2d(512,17,kernel_size=1)
model.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1)))

for param in model.classifier.parameters():
    param.requires_grad = True

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