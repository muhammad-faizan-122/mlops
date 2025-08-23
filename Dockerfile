# Use official PyTorch image as base (with CPU support)
FROM python:3.11.13

# Set working directory
WORKDIR /app

# Copy everything into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command: run inference
CMD ["python", "infer.py"]
