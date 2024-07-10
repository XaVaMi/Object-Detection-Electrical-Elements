import jetson.inference
import jetson.utils
import numpy as np
import cv2
import time
import os

# Define colors to map classes
class_colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}

# Load the object detection model with adjusted threshold
net = jetson.inference.detectNet(
    "ssd-mobilenet-v2", 
    [
        "--model=/electrical_project/ssd-mobilenet.onnx",
        "--labels=/electrical_project/labels.txt",
        "--input-blob=input_0",
        "--output-cvg=scores",
        "--output-bbox=boxes",
        "--threshold=0.5"  # Adjust the confidence threshold
    ]
)

# Load the image
image_path = "electrical_project/relay02.jpg"
if not os.path.isfile(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit(1)

image = cv2.imread(image_path)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
if image is None:
    print(f"Error: Failed to load image from {image_path}")
    exit(1)

# Convert the image to the format expected by the model (RGBA, float32)
img = jetson.utils.cudaFromNumpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).astype(np.float32))

# Start the timer for time measurement
t0 = time.time()

# Perform object detection
detections = net.Detect(img)
n_obj = len(detections)

# stop timer
delta_t = time.time() - t0
print(f"Object detection time: {delta_t:.4f} seconds")

# Process and display detections
for detect in detections:
    class_id = detect.ClassID
    item = net.GetClassDesc(class_id)
    confidence = detect.Confidence

    # get color for the detected class
    color = class_colors.get(class_id, (255, 255, 255))  # default is white

    # Create a transparent overlay for the detected object
    overlay = image.copy()
    cv2.rectangle(overlay, (int(detect.Left), int(detect.Top)), (int(detect.Right), int(detect.Bottom)), color, -1)
    alpha = 0.15  # Set the transparency level (0.0 to 1.0)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw bounding box, label, and confidence
    label = f"{item} ({confidence:.2f})"
    cv2.putText(image, label, (int(detect.Left), int(detect.Top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(image, (int(detect.Left), int(detect.Top)), (int(detect.Right), int(detect.Bottom)), color, 2)

    # Calculate and display the area of the bounding box
    box_area = (detect.Right - detect.Left) * (detect.Bottom - detect.Top)
    print(f"Detected object: {item}")
    print(f"Confidence Score: {confidence:.2f}")
    print(f"Bounding box area: {box_area:.2f}")
    print("--------------------------")

# Display the image with detections in a resizable window
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", 1920, 1080)
cv2.imshow("Object Detection", image)
cv2.waitKey(0)

# Save the result image
result_path = "electrical_project/detection_result.jpg"
cv2.imwrite(result_path, image)
print(f"Detection results saved to {result_path}")

# Clean up resources
cv2.destroyAllWindows()
