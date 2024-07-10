import jetson.inference
import jetson.utils
import numpy as np
import cv2
import time

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

# Initialize the video source (camera or video file)
camera = jetson.utils.videoSource("/dev/video0")  # '/dev/video0' for V4L2 camera, 'csi://0' for MIPI CSI camera, or the path to a video file
display = jetson.utils.videoOutput("display://0")  # 'my_video.mp4' for file output or 'display://0' for display

while display.IsStreaming():
    # Capture the image
    img = camera.Capture()

    # Start the timer for time measurement
    t0 = time.time()

    # Perform object detection
    detections = net.Detect(img)

    # Stop the timer
    delta_t = time.time() - t0
    print(f"Object detection time: {delta_t:.4f} seconds")

    # Convert the image back to numpy for display with OpenCV
    img_np = jetson.utils.cudaToNumpy(img)
    image = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR).astype(np.uint8)

    # Process and display detections
    for detect in detections:
        class_id = detect.ClassID
        item = net.GetClassDesc(class_id)
        confidence = detect.Confidence

        # Get color for the detected class
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

    # Convert back to CUDA image for display
    img_cuda = jetson.utils.cudaFromNumpy(image)

    # Render the image
    display.Render(img_cuda)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

    # Optionally, save the result image frame
    # cv2.imwrite("detection_result_frame.jpg", image)
