import torch
import torch.nn as nn
import torch.optim as optim
from models.net import Net
from utils import get_data_loaders
from logger import setup_logger

# Initialize logger
logger = setup_logger(name="train_logger", log_file="training.log")


def train_model():
    logger.info("🚀 Starting training...")

    trainloader, _ = get_data_loaders()
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):
        running_loss = 0.0
        logger.info(f"🔁 Epoch {epoch+1} started")
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                avg_loss = running_loss / 2000
                logger.info(f"[Epoch {epoch+1}, Batch {i+1}] loss: {avg_loss:.3f}")
                running_loss = 0.0

    torch.save(net.state_dict(), "cifar_net.pth")
    logger.info("✅ Model saved to cifar_net.pth")


if __name__ == "__main__":
    train_model()
