from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from models.net import Net
from utils import CIFAR_CLASSES
from logger import setup_logger
import io
import traceback

# Initialize logger
logger = setup_logger()


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


@app.get("/")
async def root():
    logger.info("📥 GET / - Health check")
    return {
        "message": "Welcome to the CIFAR-10 Classifier API. Use the /predict endpoint to classify images."
    }


def load_model():
    """Load the pre-trained model for inference.
    This function is called when the FastAPI app starts.
    """
    try:
        # Load model
        model = Net()
        model.load_state_dict(torch.load("cifar_net.pth", map_location="cpu"))
        model.eval()
        logger.info("✅ Model loaded and ready for inference.")
        return model
    # file not found
    except FileNotFoundError:
        logger.error("❌ Model file not found: cifar_net.pth")
        raise Exception("Model file not found. Please train the model first.")
    # fastapi HttpException status code 500
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to load Model")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"📥 POST /predict - Received file: {file.filename}")
        model = load_model()
        image_bytes = await file.read()

        logger.info(f"🔍 File size: {len(image_bytes)} bytes")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = inference_transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            predicted = torch.argmax(outputs, 1)
            predicted_class = CIFAR_CLASSES[predicted.item()]

        logger.info(f"✅ Prediction: {predicted_class}")
        return JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        logger.error("❌ Error during prediction")
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)
