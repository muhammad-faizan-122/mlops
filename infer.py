import torch
from PIL import Image
from torchvision import transforms
from models.net import Net
from utils import CIFAR_CLASSES
from logger import setup_logger
import argparse
import os

# Setup logger
logger = setup_logger(name="predict_logger", log_file="predict.log")


def predict_image(image_path):
    if not os.path.exists(image_path):
        logger.error(f"❌ File does not exist: {image_path}")
        return

    logger.info(f"📥 Predicting image: {image_path}")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        model = Net()
        model.load_state_dict(torch.load("cifar_net.pth", map_location="cpu"))
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = CIFAR_CLASSES[predicted.item()]

        logger.info(f"✅ Predicted class: {predicted_class}")
        print(f"🔍 Predicted class: {predicted_class}")

    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict a single image using CIFAR-10 CNN"
    )
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    predict_image(args.image_path)
