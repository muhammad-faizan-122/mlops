import torch
import torch.nn as nn
import torch.optim as optim
from models.net import Net
from utils import get_data_loaders


def train_model():
    trainloader, _ = get_data_loaders()
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch+1}, {i+1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    torch.save(net.state_dict(), "cifar_net.pth")
    print("✅ Model saved as cifar_net.pth")


if __name__ == "__main__":
    train_model()
