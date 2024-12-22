from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch import flatten
import torch.nn.functional as F
from tempfile import NamedTemporaryFile
import httpx

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 28 * 28, 1024)  # Adjust size based on image input size (224x224)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)  # Output layer for binary classification

        # Dropout layer for regularization
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x
app = Flask(__name__)

# Model loading
model = torch.load("cnn_model_full.pth")
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).cpu().item()
    label = "Malignant" if prediction > 0.5 else "Benign"
    return label, prediction

def generate_grad_cam(model, input_tensor, target_class):
    model.eval()  # Set the model to evaluation mode

    # Hook the gradients of the last convolutional layer
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks to the last convolutional layer
    target_layer = model.conv3  # Last convolutional layer
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Forward pass
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)

    # Backward pass for the target class
    model.zero_grad()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output)

    # Get gradients and activations
    grad = gradients[0].cpu().detach().numpy()
    activation = activations[0].cpu().detach().numpy()

    # Compute Grad-CAM
    weights = np.mean(grad, axis=(2, 3), keepdims=True)
    cam = np.sum(weights * activation, axis=1).squeeze()
    cam = np.maximum(cam, 0)  # ReLU operation
    cam = cam / np.max(cam)  # Normalize to [0, 1]

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return cam

def visualize_grad_cam(image, grad_cam, save_path, title="Grad-CAM"):
    """
    Visualize Grad-CAM with the image and save the result.
    """
    # Ensure image is in HWC format
    if image.shape[0] == 3:  # If image is (C, H, W), convert to (H, W, C)
        image = image.permute(1, 2, 0).numpy()

    grad_cam = np.uint8(255 * grad_cam)
    grad_cam = cv2.resize(grad_cam, (image.shape[1], image.shape[0]))

    # Overlay Grad-CAM on the image
    heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(np.array(image * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

    # Save the visualization
    cv2.imwrite(save_path, overlay)

    return save_path


@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.json
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400

        # Download the image
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            response = httpx.get(image_url)
            if response.status_code == 200:
                temp_file.write(response.content)
            else:
                return jsonify({"error": "Failed to download the image"}), 400

            # Perform prediction
            label, confidence = predict_single_image(temp_file.name)

        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/gradcam', methods=['POST'])
def gradcam_single_image():
    try:
        data = request.json
        image_url = data.get('image_url')
        target_class = data.get('target_class', 0)  # Default to class 0

        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download the image
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            response = httpx.get(image_url)
            if response.status_code == 200:
                temp_file.write(response.content)
            else:
                print(f"HTTP Error: {response.status_code}, Content: {response.content}")
                return jsonify({"error": "Failed to download the image"}), 400

            # Open and preprocess the image
            image = Image.open(temp_file.name).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Generate Grad-CAM
            grad_cam = generate_grad_cam(model, input_tensor, target_class)

            # Save Grad-CAM visualization
            save_path = temp_file.name.replace(".jpg", "_gradcam.jpg")
            visualize_grad_cam(transform(image).squeeze().cpu(), grad_cam, save_path)

            # Return the file path for download or further use
            return send_file(save_path, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
