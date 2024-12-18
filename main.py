from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import base64
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from io import BytesIO
from PIL import Image
import uuid

app = FastAPI()

UPLOAD_DIR = "./uploads"
ANNOTATED_DIR = "./annotated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

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

def analyze_symmetry_major_axis(image_path):
    # Step 1: Process the image and get the closest contour
    image_rgb, segmented_image, final_mask, annotated_image = process_image(image_path)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the closest contour
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

    # Step 2: Perform PCA on the contour points to find the major axis
    points = closest_contour[:, 0, :]  # Extract (x, y) coordinates
    mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=None)

    # Step 3: Rotate the image to align the major axis vertically
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
    rotation_matrix = cv2.getRotationMatrix2D(tuple(mean[0]), -angle, 1.0)
    rotated_mask = cv2.warpAffine(final_mask, rotation_matrix, (final_mask.shape[1], final_mask.shape[0]))

    # Step 4: Crop the rotated lesion region
    x, y, w, h = cv2.boundingRect(rotated_mask)
    cropped_mask = rotated_mask[y:y+h, x:x+w]
    cropped_lesion = image_rgb[y:y+h, x:x+w] * (cropped_mask[:, :, np.newaxis] > 0)

    # Step 5: Divide the lesion into left and flipped right halves
    mid_x = cropped_mask.shape[1] // 2
    left_half = cropped_lesion[:, :mid_x]
    right_half = cropped_lesion[:, mid_x:]
    flipped_right_half = np.flip(right_half, axis=1)

    # Step 6: Pad halves to match dimensions
    min_rows = min(left_half.shape[0], flipped_right_half.shape[0])
    max_cols = max(left_half.shape[1], flipped_right_half.shape[1])

    padded_left = np.zeros((min_rows, max_cols, 3), dtype=left_half.dtype)
    padded_right = np.zeros((min_rows, max_cols, 3), dtype=flipped_right_half.dtype)

    padded_left[:left_half.shape[0], :left_half.shape[1], :] = left_half
    padded_right[:flipped_right_half.shape[0], :flipped_right_half.shape[1], :] = flipped_right_half

    # Step 7: Compute SSIM
    ssim, _ = compare_ssim(padded_left, padded_right, full=True, multichannel=True, win_size=3)

    # Step 8: Visualize and Save Results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cropped_lesion)
    plt.title("Cropped Lesion")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(padded_left)
    plt.title("Left Half")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(padded_right)
    plt.title("Flipped Right Half")
    plt.axis("off")

    ssim_diff = np.abs(padded_left.astype(float) - padded_right.astype(float))
    plt.subplot(1, 4, 4)
    plt.imshow(ssim_diff, cmap='hot')
    plt.title(f"SSIM Difference (SSIM: {ssim:.4f})")
    plt.axis("off")

    plt.show()

    return ssim

def analyze_symmetry_updated(image_path):
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

    # Step 2: Create a mask for the closest contour
    mask = np.zeros_like(final_mask)
    cv2.drawContours(mask, [closest_contour], -1, 255, thickness=cv2.FILLED)

    # Step 3: Extract the contour's bounding box and crop
    x, y, w, h = cv2.boundingRect(closest_contour)
    cropped_mask = mask[y:y + h, x:x + w]
    cropped_lesion = image_rgb[y:y + h, x:x + w] * (cropped_mask[:, :, np.newaxis] > 0)

    # Step 4: Calculate the lesion's centroid and divide
    moments = cv2.moments(closest_contour)
    cx = int(moments["m10"] / moments["m00"]) - x
    mid_point = cx

    # Divide into left and right halves
    left_half = cropped_lesion[:, :mid_point]
    right_half = cropped_lesion[:, mid_point:]

    # Flip the right half horizontally
    flipped_right_half = np.flip(right_half, axis=1)

    # Step 5: Pad the smaller half to match dimensions
    min_rows = min(left_half.shape[0], flipped_right_half.shape[0])
    max_cols = max(left_half.shape[1], flipped_right_half.shape[1])

    padded_left = np.zeros((min_rows, max_cols, 3), dtype=left_half.dtype)
    padded_right = np.zeros((min_rows, max_cols, 3), dtype=flipped_right_half.dtype)

    padded_left[:left_half.shape[0], :left_half.shape[1], :] = left_half
    padded_right[:flipped_right_half.shape[0], :flipped_right_half.shape[1], :] = flipped_right_half

    # Step 6: Compute SSIM
    ssim, _ = compare_ssim(padded_left, padded_right, full=True, multichannel=True, win_size=3)

    # Step 7: Save visualizations
    def save_image(image, name):
        file_path = os.path.join(ANNOTATED_DIR, name)
        cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return file_path

    left_half_path = save_image(padded_left, f"left_half_{os.path.basename(image_path)}")
    flipped_right_half_path = save_image(padded_right, f"flipped_right_half_{os.path.basename(image_path)}")

    # Normalize SSIM difference map
    ssim_diff = np.abs(padded_left.astype(float) - padded_right.astype(float))
    ssim_diff = (ssim_diff - ssim_diff.min()) / (ssim_diff.max() - ssim_diff.min())  # Normalize to [0, 1]
    ssim_diff = (ssim_diff * 255).astype(np.uint8)  # Scale to [0, 255] for saving

    # Save SSIM difference map
    diff_path = os.path.join(ANNOTATED_DIR, f"ssim_diff_{os.path.basename(image_path)}")
    cv2.imwrite(diff_path, ssim_diff)

    # Return results
    return {
        "ssim_score": ssim,
        "left_half_path": left_half_path,
        "flipped_right_half_path": flipped_right_half_path,
        "ssim_diff_path": diff_path
    }

def analyze_border_and_circularity(image_path):
    # Step 1: Process the image and get the closest contour
    image_rgb, segmented_image, final_mask, annotated_image = process_image(image_path)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the closest contour (already calculated in process_image)
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

    # Save the annotated image to the annotated directory
    annotated_file_path = os.path.join(ANNOTATED_DIR, os.path.basename(image_path))
    cv2.imwrite(annotated_file_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Step 4: Save visualizations as base64
    def save_image(image, title):
        buffer = BytesIO()
        plt.figure()
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.savefig(buffer, format="png", bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    output_images = {}
    output_images['segmented_image'] = save_image(segmented_image, "Segmented Image")
    output_images['annotated_image'] = save_image(annotated_image, "Annotated Image")

    # Return metrics and visualizations
    return {
        "border_irregularity": border_irregularity,
        "circularity": circularity,
        "annotated_file_path": annotated_file_path,  # Add this for debugging
        "images": output_images
    }

def analyze_evolution(image_path1, image_path2):
    # Process images
    _, _, mask1, _ = process_image(image_path1)
    _, _, mask2, _ = process_image(image_path2)

    # Resize masks
    min_rows = min(mask1.shape[0], mask2.shape[0])
    min_cols = min(mask1.shape[1], mask2.shape[1])
    resized_mask1 = cv2.resize(mask1, (min_cols, min_rows), interpolation=cv2.INTER_NEAREST)
    resized_mask2 = cv2.resize(mask2, (min_cols, min_rows), interpolation=cv2.INTER_NEAREST)

    # Compute SSIM
    ssim, diff = compare_ssim(resized_mask1, resized_mask2, full=True)

    # Normalize the difference map for visualization
    diff_normalized = (diff - diff.min()) / (diff.max() - diff.min()) * 255
    diff_colormap = cv2.applyColorMap(diff_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # Save results
    unique_name1 = os.path.basename(image_path1)
    unique_name2 = os.path.basename(image_path2)
    mask1_path = os.path.join(ANNOTATED_DIR, f"mask1_{unique_name1}")
    mask2_path = os.path.join(ANNOTATED_DIR, f"mask2_{unique_name2}")
    diff_path = os.path.join(ANNOTATED_DIR, f"ssim_diff_{unique_name1}_{unique_name2}")

    cv2.imwrite(mask1_path, resized_mask1 * 255)
    cv2.imwrite(mask2_path, resized_mask2 * 255)
    cv2.imwrite(diff_path, diff_colormap)

    # Return paths and SSIM score
    return {
        "ssim_score": ssim,
        "mask1_path": mask1_path,
        "mask2_path": mask2_path,
        "ssim_diff_path": diff_path,
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": "Image uploaded successfully", "file_path": file_path}

@app.post("/analyze-asymmetry")
async def analyze_asymmetry(file: UploadFile = File(...)):
    # Save the uploaded file
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Perform asymmetry analysis
    try:
        results = analyze_symmetry_major_axis(file_path)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/analyze-border-circularity")
async def analyze_border_circularity(file: UploadFile = File(...)):
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Perform the border and circularity analysis
    try:
        results = analyze_border_and_circularity(file_path)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/analyze-evolution")
async def analyze_evolution_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Save the first uploaded file
    file1_path = os.path.join(UPLOAD_DIR, file1.filename)
    with open(file1_path, "wb") as f:
        f.write(await file1.read())  # Corrected 'file' to 'file1'

    # Save the second uploaded file
    file2_path = os.path.join(UPLOAD_DIR, file2.filename)
    with open(file2_path, "wb") as f:
        f.write(await file2.read())  # Corrected 'file' to 'file2'

    # Perform the evolution analysis
    try:
        results = analyze_evolution(file1_path, file2_path)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)