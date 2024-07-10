
# Object Detection of Electrical Elements
## Description of the project
This project aims to detect objects in electrical installations and high-voltage switchboards, specifically:

* Electrical components: 2 and 3-pole circuit breakers and relays.
![classes](https://github.com/XaVaMi/Object-Detection-Electrical-Elements/assets/173626888/991b5212-f5de-472a-899b-36f28bbd6af4)

* Description:
The project utilizes a Jetson Nano board, following tutorials from [Hello AI World](https://github.com/dusty-nv/jetson-inference), and relies on the MobilNet v2 and SSD machine learning models to detect the target objects with accuracy and efficiency. In order to train the model, repository files for training and evaluation are modified from [Pytorch SSD](https://github.com/qfgaohao/pytorch-ssd). The labelling and annotation of the images has been done with [LabelMe](https://github.com/labelmeai/labelme) and the conversion of the json annotations to XML for training with [Roboflow](https://roboflow.com/convert/labelbox-json-to-pascal-voc-xml).

* Features:
Real-time detection of electrical components in images and videos.
Lightweight and efficient machine learning model, ideal for running on Jetson Nano.
High accuracy and detection rates.
Easy to use and customize.

* Prerequisites:
Jetson Nano board with
NVIDIA JetPack operating system
OpenCV
TensorFlow
Pytorch. Webcam.

* Installation:
Clone this repository to your Jetson Nano.
Install the required dependencies:
OpenCV
TensorFlow
Download the pre-trained MobilNet v2 and SSD machine learning models.
Run the object detection script: python [`electrical_detect.py`]()

* Usage:
The object detection script will display a live feed from the Jetson Nano's camera.
Detection from image and video resources is also possible.
Detected electrical components will be highlighted with bounding boxes and labels and saved in a csv file.

## Data augmentation tools
**Simple augmentation** with
[`data_aug_electrical.py`](https://github.com/XaVaMi/Object-Detection-Electrical-Elements/blob/main/Data%20augmentation%20tools/data_aug_electrical.py) file. 

If you have many annotations of a class, in order not to further aggravate the imbalance problem, you can select the class to be eliminated and apply the effects of augmentation to the image to increase the number of annotations of lesser representation, ignoring and not increasing the selected one. If no class is selected, it applies the effects to the entire image. The effects generated to the images and annotations are grayscale, shear, scale and rotation.

![Captura de pantalla 2024-06-24 a les 4 50 45](https://github.com/XaVaMi/Object-Detection-Electrical-Elements/assets/173626888/7c12a6bf-ed1d-41a8-bd62-a67846f3de6c)

**Creating a mosaic for augmentation** with 
[`mosaic_generation.py`](https://github.com/XaVaMi/Object-Detection-Electrical-Elements/blob/main/Data%20augmentation%20tools/mosaic_generation.py) file. 

The idea of ​​this code is for a class of which there is little representation, from a label in an image, to obtain multiple new copies to expand the number of this category. Mainly these generated mosaics should be collected in the train subset since, although we provide our model with many examples to learn during training, it will easily find them on the black background. To challenge the model, let's use real images, expanded with techniques to expand our dataset or not, and let's make it work!

![mosaic_explanation](https://github.com/XaVaMi/Object-Detection-Electrical-Elements/assets/173626888/a2795105-a41f-44d6-be4d-efdb34a8be3a)

Similar to the previous code we have [`Mosaic_Augmentation.ipynb`](https://github.com/XaVaMi/Object-Detection-Electrical-Elements/blob/main/Data%20augmentation%20tools/Mosaic_Augmentation.ipynb) which is a Jupyter notebook. We can access our Drive files, select the classes to which we want to apply the mosaic and different effects. It can be applied only to those that are desired and the results can be viewed.

> [!NOTE]
> These augmentation code files are prepared for LabelMe json files annotations.

## Electrical Elements with dataset v01
The following figure presents the diagram of the steps carried out.
![general_scheme](https://github.com/XaVaMi/Object-Detection-Electrical-Elements/assets/173626888/6625cdb9-a129-44cc-a067-30e8cbd35698)

> [!WARNING]
> The distribution and division for creating the subsets is 80% for train, 10% for valid and 10% for test.

## Object detection examples
In this video there is a small explanation of the motivation of the project and how it was born, as well as the process to bring it to completion. Some execution examples are also included in the brief report.

