# 🧠 MLOps - Image Classifier Deployment with Minikube

The goal of this project is to **build and deploy an image classifier API** using Docker and Minikube (a local Kubernetes environment).

---

## 🚀 Quick Test

You can send an API request to the application running inside Minikube.
Example screenshot:

![Demo](test/demo1.png)

---

## 🛠️ Setup & Run Instructions

### ✅ Prerequisites

1. **Install Docker:**
   Follow official instructions for your OS:
   👉 [Docker Installation Guide](https://docs.docker.com/engine/install/)

2. **Install Minikube:**
   Follow instructions based on your OS:
   👉 [Minikube Installation Guide](https://minikube.sigs.k8s.io/docs/start/?arch=%2Flinux%2Fx86-64%2Fstable%2Fbinary+download)

---

### 🧱 Step-by-Step Guide

#### 1. Build the Docker Image

```bash
docker build -t classifier:v1 .
```

#### 2. Load the Docker Image into Minikube

```bash
minikube image load classifier:v1
```

📌 You can also follow this [medium article](https://medium.com/codex/how-to-use-local-docker-images-with-minikube-6dcf0ef05a2d) for further explanation.

#### 3. Confirm Image Exists in Minikube

```bash
minikube ssh
docker images
```

Look for `classifier:v1` in the list.

#### 4. Start Minikube

```bash
minikube start
```

#### 5. Apply Kubernetes Deployment & Service

```bash
kubectl apply -f cnn-deployment.yaml
kubectl apply -f cnn-service.yaml
```

#### 6. Run the Service

```bash
minikube service cnn-fastapi-service
```

You will see output like this:

```
|-----------|---------------------|-------------|---------------------------|
| NAMESPACE |        NAME         | TARGET PORT |            URL            |
|-----------|---------------------|-------------|---------------------------|
| default   | cnn-fastapi-service |          80 | http://192.168.49.2:30007 |
|-----------|---------------------|-------------|---------------------------|
🎉  Opening service default/cnn-fastapi-service in default browser...
```

You can now make requests to the given URL to test your classifier!

---

## 📂 Project Folder Structure

```
.
├── cifar_net.pth                # Trained model weights
├── cnn-deployment.yaml          # Kubernetes deployment file
├── cnn-service.yaml             # Kubernetes service file
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies

├── main.py                      # Entry point for FastAPI app
├── infer.py                     # Image inference script
├── train.py                     # Training script for model
├── utils.py                     # Helper functions
├── logger.py                    # Logging setup
├── READMe.md                    # Project documentation

├── models/
│   ├── net.py                   # CNN model definition
│   └── __pycache__/             # Compiled Python files

├── experiments/
│   └── test.ipynb               # Jupyter notebook for testing

├── logs/
│   ├── inference.log            # Logs related to inference
│   └── utils.log                # Logs for utility functions

├── test/
│   ├── demo1.png                # Screenshot of API test
│   ├── dog.jpeg                 # Sample test image
│   └── plane.jpg                # Sample test image

├── __pycache__/                 # Compiled Python cache
│   ├── logger.cpython-310.pyc
│   ├── main.cpython-310.pyc
│   └── utils.cpython-310.pyc
```

---

## 🧪 Example API Request (Optional Section)

If you have an example cURL or Python request, add it here:

```bash
curl -X POST http://<MINIKUBE-IP>:<PORT>/predict \
    -F "file=@path_to_image.jpg"
```

---