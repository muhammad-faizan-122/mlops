import torch
from PIL import Image
from torchvision import transforms
from models.net import Net
from utils import CIFAR_CLASSES


def predict_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    model = Net()
    model.load_state_dict(torch.load("cifar_net.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        print("🔍 Predicted class:", CIFAR_CLASSES[predicted.item()])


if __name__ == "__main__":
    predict_image("test/image.jpg")  # Replace with your image path
