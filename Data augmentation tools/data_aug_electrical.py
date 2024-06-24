import os
import cv2
import numpy as np
import json

# --- Definition of auxiliary functions ---

def apply_rotation(image, shapes, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the bounding box of the rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

    rotated_shapes = []
    for shape in shapes:
        rotated_points = []
        for point in shape['points']:
            new_point = np.dot(M, np.array([point[0], point[1], 1]))
            rotated_points.append(new_point.tolist())
        rotated_shape = shape.copy()
        rotated_shape['points'] = rotated_points
        rotated_shapes.append(rotated_shape)

    return rotated_image, rotated_shapes

def scale_image(image, shapes, scale_factor):
    (h, w) = image.shape[:2]
    scaled_image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))

    scaled_shapes = []
    for shape in shapes:
        scaled_points = [[point[0] * scale_factor, point[1] * scale_factor] for point in shape['points']]
        scaled_shape = shape.copy()
        scaled_shape['points'] = scaled_points
        scaled_shapes.append(scaled_shape)
    return scaled_image, scaled_shapes

def apply_shear(image, shapes, shear_factor, direction='x'):
    (h, w) = image.shape[:2]
    if direction == 'x':
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        new_w = w + int(shear_factor * h)
        new_h = h
    elif direction == 'y':
        M = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
        new_w = w
        new_h = h + int(shear_factor * w)
    
    sheared_image = cv2.warpAffine(image, M, (new_w, new_h))

    sheared_shapes = []
    for shape in shapes:
        sheared_points = []
        for point in shape['points']:
            if direction == 'x':
                new_point = [point[0] + shear_factor * point[1], point[1]]
            elif direction == 'y':
                new_point = [point[0], point[1] + shear_factor * point[0]]
            sheared_points.append(new_point)
        sheared_shape = shape.copy()
        sheared_shape['points'] = sheared_points
        sheared_shapes.append(sheared_shape)

    return sheared_image, sheared_shapes

def apply_grayscale(image, shapes):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR) # Convert back to 3 channels
    return grayscale_image, shapes

def mask_selected_areas(image, shapes, class_list):
    mask = np.zeros_like(image)
    for shape in shapes:
        if shape['label'] in class_list:
            points = np.array(shape['points'], np.int32)
            cv2.fillPoly(mask, [points], (255, 255, 255))
    masked_image = cv2.bitwise_and(image, cv2.bitwise_not(mask))
    new_shapes = [shape for shape in shapes if shape['label'] not in class_list]
    return masked_image, new_shapes

def augment_data(input_directory, output_directory, angles, scale_factors, shear_factors, class_list):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            annotation_path = os.path.join(input_directory, filename)
            with open(annotation_path, 'r') as f:
                annotation_data = json.load(f)
            image_path = os.path.join(input_directory, annotation_data['imagePath'])
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                print(f"Warning: Unable to load image at path: {image_path}")
                continue
            
            masked_image, filtered_shapes = mask_selected_areas(original_image, annotation_data['shapes'], class_list)

            # Apply transformations to the masked image and filtered shapes
            for suffix, transform_func, param in [
                ('rotated', apply_rotation, angles),
                ('scaled', scale_image, scale_factors),
                ('sheared_x', lambda img, shapes, shear: apply_shear(img, shapes, shear, direction='x'), shear_factors),
                ('sheared_y', lambda img, shapes, shear: apply_shear(img, shapes, shear, direction='y'), shear_factors),
                ('grayscale', apply_grayscale, [None])
            ]:
                for p in param:
                    if p is None:
                        aug_image, aug_shapes = transform_func(masked_image, filtered_shapes)
                    else:
                        aug_image, aug_shapes = transform_func(masked_image, filtered_shapes, p)
                    
                    # Save the augmented image and its annotation
                    image_filename = f"{os.path.basename(image_path).split('.')[0]}_{suffix}_{p if p is not None else ''}.jpg"
                    image_output_path = os.path.join(output_directory, image_filename)
                    cv2.imwrite(image_output_path, aug_image)

                    new_annotation_data = {
                        'version': annotation_data['version'],
                        'flags': annotation_data['flags'],
                        'shapes': aug_shapes,
                        'imagePath': image_filename,
                        'imageHeight': aug_image.shape[0],
                        'imageWidth': aug_image.shape[1],
                        'imageData': None
                    }
                    annotation_filename = f"{os.path.basename(image_path).split('.')[0]}_{suffix}_{p if p is not None else ''}.json"
                    annotation_output_path = os.path.join(output_directory, annotation_filename)
                    with open(annotation_output_path, 'w') as f:
                        json.dump(new_annotation_data, f, indent=4)

    print("Data augmentation process finished.")

# --- PARAMETERS ---
input_directory = '/Users/Xavier/Desktop/dataset_02'
output_directory = '/Users/Xavier/Desktop/dataset_02/aug'
angles = [20, -20]
scale_factors = [0.5, 1.5]
shear_factors = [0.2]
class_list = []# no class selected applies effects to all images and masks

augment_data(input_directory, output_directory, angles, scale_factors, shear_factors, class_list)
