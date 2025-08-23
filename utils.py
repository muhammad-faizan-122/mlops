import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from logger import setup_logger

# Setup logger
logger = setup_logger(name="utils_logger", log_file="utils.log")

# CIFAR-10 class names
CIFAR_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_data_loaders(batch_size=4):
    try:
        logger.info("📦 Preparing CIFAR-10 data loaders...")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        logger.info(f"✅ Training data loaded. Samples: {len(trainset)}")

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        logger.info(f"✅ Test data loaded. Samples: {len(testset)}")

        return trainloader, testloader

    except Exception as e:
        logger.error("❌ Error in get_data_loaders()")
        logger.exception(e)
        raise


def imshow(img):
    try:
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        logger.info("🖼️ Image displayed using imshow()")
    except Exception as e:
        logger.error("❌ Failed to display image in imshow()")
        logger.exception(e)
