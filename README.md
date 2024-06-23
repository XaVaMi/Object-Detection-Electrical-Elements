
# Object Detection Electrical Elements
This project aims to detect objects in electrical installations and high-voltage switchboards, specifically:

* Electrical components: 2 and 3-pole circuit breakers and relays.

* Description:
The project utilizes a Jetson Nano board and relies on the MobilNet v2 and SSD machine learning models to detect the target objects with accuracy and efficiency.

* Features:
Real-time detection of electrical components in images and videos.
Lightweight and efficient machine learning model, ideal for running on Jetson Nano.
High accuracy and detection rates.
Easy to use and customize.

* Prerequisites:
Jetson Nano board
NVIDIA JetPack operating system
OpenCV
TensorFlow

* Installation:
Clone this repository to your Jetson Nano.
Install the required dependencies:
OpenCV
TensorFlow
Download the pre-trained MobilNet v2 and SSD machine learning models.
Run the object detection script: python electrical_detect.py

* Usage:
The object detection script will display a live feed from the Jetson Nano's camera.
Detection from image and video resources is also possible.
Detected electrical components will be highlighted with bounding boxes and labels and saved in a csv file.
