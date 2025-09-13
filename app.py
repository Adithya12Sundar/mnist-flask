

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import os



class samplefd(nn.Module):
  def __init__(self):
    super(samplefd, self).__init__()
    self.block = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels = 3, kernel_size=3,stride=1,padding = 0),
    nn.MaxPool2d(kernel_size = 2, stride = 1),
    nn.Conv2d(in_channels = 3, out_channels=3, kernel_size = 2),
    nn.Conv2d(in_channels=3, out_channels=4,kernel_size = 2),
    nn.MaxPool2d(kernel_size=2, stride = 1),
    nn.Flatten(),
    nn.Linear(in_features = 4*22*22, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features = 128, out_features = 64),
    nn.Linear(in_features = 64, out_features = 28),nn.ReLU(),
    nn.Linear(in_features = 28, out_features = 10),nn.ReLU())
  def forward(self, x):
    return self.block(x)


# ---- Load model ----
model_path = os.path.join("models", "mnist_model.pth")
model = CNN()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ---- Flask app ----
app = Flask(__name__)

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/")
def home():
    return jsonify({"message": "MNIST Flask API is live!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("L")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


