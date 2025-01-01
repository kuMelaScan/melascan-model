from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch import flatten
import torch.nn.functional as F
from tempfile import NamedTemporaryFile
import httpx
from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import base64
import cv2
from io import BytesIO
from PIL import Image
import uuid
from flask import send_file
from skimage import filters
from skimage.color import rgb2gray
import base64


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

def get_cropped_lesion(image):
    """
    Extract the cropped lesion from the input image using segmentation and contour logic.

    Args:
        image (np.ndarray): Input image as a NumPy array.

    Returns:
        PIL.Image: The cropped lesion as a PIL Image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    # Convert to RGB if the input is not already in RGB format
    if len(image.shape) == 2 or image.shape[2] != 3:  # Grayscale or other format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image

    # Use your `segment_lesion` function to segment the lesion
    segmented_image, final_mask = segment_lesion(image_rgb)

    # Find contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the image center
    image_center = (image.shape[1] // 2, image.shape[0] // 2)  # (cx, cy)

    # Find the closest contour
    closest_contour = None
    min_distance = float('inf')
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:  # Avoid division by zero
            cx_contour = int(moments["m10"] / moments["m00"])
            cy_contour = int(moments["m01"] / moments["m00"])
            distance = np.sqrt((cx_contour - image_center[0]) ** 2 + (cy_contour - image_center[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

    if closest_contour is None:
        raise ValueError("No valid contour found near the center.")

    # Extract the bounding box of the closest contour
    x, y, w, h = cv2.boundingRect(closest_contour)

    # Create a mask for the contour
    mask = np.zeros_like(final_mask)
    cv2.drawContours(mask, [closest_contour], -1, 255, thickness=cv2.FILLED)

    # Crop the lesion from the image
    cropped_mask = mask[y:y + h, x:x + w]
    cropped_lesion = image_rgb[y:y + h, x:x + w] * (cropped_mask[:, :, np.newaxis] > 0)

    # Convert the cropped lesion to a PIL image
    cropped_lesion_pil = Image.fromarray(cropped_lesion.astype('uint8'))

    return cropped_lesion_pil

def predict_single_image(image):
    
    # Step 1: Apply transformations and prepare input tensor
    input_tensor = transform(Image.fromarray(image)).unsqueeze(0)  # Transform the full image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    # Step 2: Perform prediction using the model
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).cpu().item()

    # Step 3: Determine the label based on the prediction score
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
def remove_hair_improved(image):
    """
    Hair removal method with reduced blurriness.
    :param image: Input image in BGR format
    :return: Image with hair removed and less blurriness
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a BlackHat filter to detect hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # Smaller kernel size for finer hair detection
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold the blackhat result to create a mask of hair
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the image to remove the hair with a smaller inpainting radius
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    return inpainted_image
    
def segment_lesion(image):
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a mask for GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[thresh == 255] = cv2.GC_FGD  # Foreground
    mask[thresh == 0] = cv2.GC_BGD    # Background

    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut with the mask
    cv2.grabCut(image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    # Extract the segmented region
    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * final_mask[:, :, np.newaxis]

    return segmented_image, final_mask

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Cannot load image at {image_path}. Check the file path or integrity.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Segment the lesion
    segmented_image, final_mask = segment_lesion(image)

    # Find contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the image center
    image_center = (image.shape[1] // 2, image.shape[0] // 2)  # (cx, cy)

    # Initialize variables to store the closest contour
    closest_contour = None
    min_distance = float('inf')

    # Iterate through all contours
    for contour in contours:
        # Calculate the centroid of the contour
        moments = cv2.moments(contour)
        if moments["m00"] != 0:  # Avoid division by zero
            cx_contour = int(moments["m10"] / moments["m00"])
            cy_contour = int(moments["m01"] / moments["m00"])
            # Calculate distance to the image center
            distance = np.sqrt((cx_contour - image_center[0]) ** 2 + (cy_contour - image_center[1]) ** 2)
            # Update the closest contour if this distance is smaller
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

    # Proceed if we found a valid contour
    if closest_contour is not None:
        # Compute bounding box and metrics for the closest contour
        x, y, w, h = cv2.boundingRect(closest_contour)
        contour_area = cv2.contourArea(closest_contour)
        bounding_area = w * h
        irregularity = contour_area / bounding_area
        perimeter = cv2.arcLength(closest_contour, True)
        circularity = (4 * np.pi * contour_area) / (perimeter ** 2)

        # Annotate the image
        annotated_image = image_rgb.copy()
        cv2.drawContours(annotated_image, [closest_contour], -1, (0, 255, 0), 2)  # Green contour
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red bounding box
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(annotated_image, f"Irregularity: {irregularity:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(annotated_image, f"Circularity: {circularity:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        # Return processed data
        return image_rgb, segmented_image, final_mask, annotated_image
    else:
        raise ValueError("No valid contour found near the center.")

def analyze_symmetry_updated(image_path):
    # Step 1: Process the image and find the largest contour
    image_rgb, segmented_image, final_mask, _ = process_image(image_path)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour (instead of the closest)
    if len(contours) == 0:
        raise ValueError("No valid contours found.")
    largest_contour = max(contours, key=cv2.contourArea)

    # Step 2: Perform PCA on the contour points to find the major axis
    points = largest_contour[:, 0, :].astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    major_axis = eigenvectors[0]

    # Step 3: Compute the rotation angle to align the major axis horizontally
    angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
    rotation_matrix = cv2.getRotationMatrix2D(center=tuple(mean[0]), angle=-angle, scale=1.0)

    # Step 4: Rotate the mask and the original image
    h, w = final_mask.shape
    rotated_mask = cv2.warpAffine(final_mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (image_rgb.shape[1], image_rgb.shape[0]))

    # Step 5: Crop the rotated lesion region
    contours_rotated, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_rotated) == 0:
        raise ValueError("No valid contours found after rotation.")
    largest_rotated_contour = max(contours_rotated, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest_rotated_contour)
    cropped_mask = rotated_mask[y:y+h_box, x:x+w_box]
    cropped_lesion = rotated_image[y:y+h_box, x:x+w_box] * (cropped_mask[:, :, np.newaxis] > 0)

    # Step 6: Split the lesion into left and right halves
    mid_x = cropped_mask.shape[1] // 2
    left_half = cropped_lesion[:, :mid_x]
    right_half = cropped_lesion[:, mid_x:]
    flipped_right_half = np.flip(right_half, axis=1)

    # Step 7: Pad halves to match dimensions, then compute SSIM
    min_rows = min(left_half.shape[0], flipped_right_half.shape[0])
    max_cols = max(left_half.shape[1], flipped_right_half.shape[1])
    padded_left = np.zeros((min_rows, max_cols, 3), dtype=left_half.dtype)
    padded_right = np.zeros((min_rows, max_cols, 3), dtype=flipped_right_half.dtype)
    padded_left[:left_half.shape[0], :left_half.shape[1], :] = left_half
    padded_right[:flipped_right_half.shape[0], :flipped_right_half.shape[1], :] = flipped_right_half

    ssim, _ = compare_ssim(padded_left, padded_right, full=True, multichannel=True, win_size=3)
    return {"ssim_score": ssim}

def analyze_border_and_circularity(image_path):
    # Step 1: Process the image and get the closest contour
    image_rgb, segmented_image, final_mask, annotated_image = process_image(image_path)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the closest contour
    closest_contour = None
    image_center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
    min_distance = float('inf')

    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx_contour = int(moments["m10"] / moments["m00"])
            cy_contour = int(moments["m01"] / moments["m00"])
            distance = np.sqrt((cx_contour - image_center[0]) ** 2 + (cy_contour - image_center[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

    if closest_contour is None:
        raise ValueError("No valid contour found near the center.")

    # Step 2: Calculate Border Irregularity and Circularity
    x, y, w, h = cv2.boundingRect(closest_contour)
    contour_area = cv2.contourArea(closest_contour)
    bounding_box_area = w * h
    perimeter = cv2.arcLength(closest_contour, True)

    # Border Irregularity
    border_irregularity = contour_area / bounding_box_area if bounding_box_area != 0 else 0

    # Circularity
    circularity = (4 * np.pi * contour_area) / (perimeter ** 2) if perimeter > 0 else 0

    # Step 3: Annotate the image with results
    cv2.drawContours(annotated_image, [closest_contour], -1, (0, 255, 0), 2)  # Green contour
    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red bounding box

    font_scale = 0.5
    font_thickness = 1
    cv2.putText(annotated_image, f"Irregularity: {border_irregularity:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(annotated_image, f"Circularity: {circularity:.2f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # Step 4: Save the annotated image
    annotated_dir = "annotated_results"
    os.makedirs(annotated_dir, exist_ok=True)
    annotated_file_path = os.path.join(annotated_dir, f"annotated_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(annotated_file_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Step 5: Return metrics and the annotated image path
    return {
        "border_irregularity": border_irregularity,
        "circularity": circularity,
        "annotated_file_path": annotated_file_path  # Include the saved file path
    }

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

def detect_pigmentation(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Cannot load image. Please check the image path.")

    # Step 1: Extract the cropped lesion
    cropped_lesion = get_cropped_lesion(image)

    # Step 2: Convert to NumPy array
    cropped_lesion_np = np.array(cropped_lesion)

    # Step 3: Grayscale conversion
    gray = cv2.cvtColor(cropped_lesion_np, cv2.COLOR_RGB2GRAY)

    # Step 4: Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 5: Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Generate heatmap
    heatmap = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Step 7: Detect edges
    edges = filters.sobel(rgb2gray(cropped_lesion_np))
    edge_overlay = cropped_lesion_np.copy()
    edge_overlay[edges > 0.1] = [255, 0, 0]

    # Save results
    results_dir = "pigmentation_results"
    os.makedirs(results_dir, exist_ok=True)

    heatmap_path = os.path.join(results_dir, f"heatmap_{uuid.uuid4().hex}.jpg")
    edge_overlay_path = os.path.join(results_dir, f"edge_overlay_{uuid.uuid4().hex}.jpg")

    # Validate outputs before saving
    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(edge_overlay_path, cv2.cvtColor(edge_overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))

    return heatmap_path, edge_overlay_path


@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.json
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400

        # Download the image
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            response = httpx.get(image_url, follow_redirects=True)
            if response.status_code == 200:
                temp_file.write(response.content)
                temp_file.close()  # Ensure the file is saved before reading it
                # Read the image as a NumPy array
                image = cv2.imread(temp_file.name)
                if image is None:
                    return jsonify({"error": "Failed to load image into NumPy array"}), 400
            else:
                return jsonify({"error": "Failed to download the image"}), 400

            # Perform prediction
            label, confidence = predict_single_image(image)

        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/gradcam', methods=['POST'])
def gradcam_single_image():
    try:
        data = request.json
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 1: Download the image
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            response = httpx.get(image_url)
            if response.status_code == 200:
                temp_file.write(response.content)
            else:
                print(f"HTTP Error: {response.status_code}, Content: {response.content}")
                return jsonify({"error": "Failed to download the image"}), 400

            image = cv2.imread(temp_file.name)
            if image is None:
                return jsonify({"error": "Failed to load image into NumPy array"}), 400

            # Step 3: Extract cropped lesion
            cropped_lesion = get_cropped_lesion(image)

            # Step 4: Preprocess the cropped lesion
            input_tensor = transform(cropped_lesion).unsqueeze(0).to(device)

            # Step 5: Predict the target class using the model
            with torch.no_grad():
                output = model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()  # Get the predicted class index

            # Step 6: Generate Grad-CAM
            grad_cam = generate_grad_cam(model, input_tensor, target_class)

            # Step 7: Save Grad-CAM visualization
            save_path = temp_file.name.replace(".jpg", "_gradcam.jpg")
            visualize_grad_cam(transform(cropped_lesion).squeeze().cpu(), grad_cam, save_path)

            # Step 8: Return the saved Grad-CAM image
            return send_file(save_path, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/detect_pigmentation', methods=['POST'])
def detect_pigmentation_route():
    try:
        # Parse the JSON request to get the image URL
        data = request.json
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400

        # Download the image
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            response = httpx.get(image_url, follow_redirects=True)
            if response.status_code == 200:
                temp_file.write(response.content)
            else:
                return jsonify({"error": f"Failed to download the image. HTTP Status: {response.status_code}"}), 400

            # Perform pigmentation detection
            heatmap_path, edge_overlay_path = detect_pigmentation(temp_file.name)

        # Return the heatmap as a downloadable file
        return send_file(
            heatmap_path,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name="heatmap.jpg"
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_symmetry', methods=['POST'])
def analyze_symmetry():
    try:
        # Parse the JSON request to get the image URL
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

            # Call the analyze_symmetry_updated function
            result = analyze_symmetry_updated(temp_file.name)

        # Extract the SSIM score
        ssim_score = result['ssim_score']

        # Build a descriptive message with a brief explanation
        if ssim_score > 0.9:
            explanation = "The lesion shows high symmetry, suggesting it is more likely benign."
        elif 0.7 <= ssim_score <= 0.9:
            explanation = "The lesion has moderate symmetry, which may require further investigation."
        else:
            explanation = "The lesion shows low symmetry, suggesting it could be malignant and warrants further evaluation."

        response_message = (
            f"The SSIM score is {ssim_score:.2f}. {explanation}"
        )

        return jsonify({"message": response_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze_border_circularity', methods=['POST'])
def analyze_border_circularity_route():
    try:
        # Parse the JSON request to get the image URL
        data = request.json
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({"error": "Image URL is required"}), 400

        # Step 1: Download the image
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            response = httpx.get(image_url)
            if response.status_code == 200:
                temp_file.write(response.content)
            else:
                return jsonify({"error": "Failed to download the image"}), 400

            # Load the image and apply hair removal
            image = cv2.imread(temp_file.name)
            if image is None:
                return jsonify({"error": "Failed to load image"}), 400

            hair_removed_image = remove_hair_improved(image)

            # Save the hair-removed image to a temporary path
            hair_removed_path = temp_file.name.replace(".jpg", "_hair_removed.jpg")
            cv2.imwrite(hair_removed_path, hair_removed_image)

        # Step 2: Pass the hair-removed image path to analyze_border_and_circularity
        result = analyze_border_and_circularity(hair_removed_path)

        # Extract metrics
        border_irregularity = result["border_irregularity"]
        circularity = result["circularity"]

        # Load the annotated image and encode it in Base64
        annotated_image_path = result["annotated_file_path"]
        with open(annotated_image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')

        # Return metrics and the image as Base64
        return jsonify({
            "border_irregularity": border_irregularity,
            "circularity": circularity,
            "annotated_image_base64": f"data:image/jpeg;base64,{base64_image}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
