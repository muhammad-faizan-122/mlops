from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from models.net import Net
from utils import CIFAR_CLASSES
import io

# Load model
model = Net()
model.load_state_dict(torch.load("cifar_net.pth", map_location="cpu"))
model.eval()

# FastAPI app
app = FastAPI(title="CIFAR-10 Classifier")

# Inference transform
inference_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = inference_transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            predicted = torch.argmax(outputs, 1)
            predicted_class = CIFAR_CLASSES[predicted.item()]

        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
