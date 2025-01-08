import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import argparse

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # After block3, the spatial size is 224 -> 112 -> 56 -> 28
        # so the feature map is (128, 28, 28) = 128 * 28 * 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)  # single output for BCEWithLogitsLoss

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)            # No sigmoid here
        return x

def load_model_state_dict(model_class, path='model_state_dict.pth', device='cpu'):
    """
    Loads a model's state dictionary and applies it to an instance of the model class.
    Args:
        model_class: The class of the model to instantiate.
        path (str): Path to the saved state dictionary.
        device (str): The device to load the model onto ('cpu' or 'cuda').
    Returns:
        model: The loaded model with weights applied.
    """
    import torch
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model state dictionary loaded from {path}")
    return model

def evaluate_model(model, image_path, transform, device='cpu'):
    """
    Evaluates the model on the first 20 images in the given directory.
    Args:
        model: The PyTorch model to evaluate.
        image_dir: Path to the directory containing images.
        transform: Transformations to apply to the images.
        device: Device to use for evaluation ('cpu' or 'cuda').
    Returns:
        results: List of tuples with filenames and predictions.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(input_tensor)  # Raw logits
        print(f"Raw model output (logits): {logits}")
        probability = torch.sigmoid(logits).item()  # Convert logits to probability
        print(f"Probability: {probability}")
    return probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single image with DeeperCNN.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("--model_path", type=str, default="new_saved_model.pth", help="Path to the saved model file.")
    args = parser.parse_args()

    # Define the transform pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model_state_dict(DeeperCNN, path=args.model_path, device=device)

    # Evaluate the provided image
    probability = evaluate_model(model, args.image_path, transform, device=device)

    # Determine label
    label = "Malignant" if probability > 0.5 else "Benign"
    print(f"Image: {args.image_path}")
    print(f"Label: {label}")
    print(f"Confidence: {probability:.4f}")