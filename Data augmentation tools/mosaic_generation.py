import os
import cv2
import numpy as np
import json
import math

# --- DEFINICIÓ FUNCIONS AUXILIARS ---

def apply_rotation(image, points, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2) # agafem centre per aplicar rotacio
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Rotate points - fer coincidir rot imatge i etiquetes (corregit)
    rotated_points = []
    for point in points:
        new_point = np.dot(M, np.array([point[0], point[1], 1]))
        rotated_points.append(new_point.tolist())

    return rotated_image, rotated_points

def scale_points(points, scale_factor):
    return [[point[0] * scale_factor, point[1] * scale_factor] for point in points]

def apply_scale(image, points, scale_factor):
    (h, w) = image.shape[:2]
    scaled_image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor))) # proporció mesures (verificat i ok)
    scaled_points = scale_points(points, scale_factor)
    return scaled_image, scaled_points

def shear_points(points, shear_factor, direction='x'):
    if direction == 'x':
        return [[point[0] + shear_factor * point[1], point[1]] for point in points]
    elif direction == 'y':
        return [[point[0], point[1] + shear_factor * point[0]] for point in points]

def apply_shear(image, points, shear_factor, direction='x'):
    (h, w) = image.shape[:2]
    if direction == 'x':
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])      # data types https://numpy.org/doc/stable/user/basics.types.html
    elif direction == 'y':
        M = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (w, h))      # més còmode mètode affine https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
    sheared_points = shear_points(points, shear_factor, direction)
    return sheared_image, sheared_points

def extract_region(image, shape, margin_ratio=0.05):
    points = np.array(shape['points'], np.int32)
    x, y, w, h = cv2.boundingRect(points) # agafem regio d'interés ROI a partir de contorns
    
    # marge afegit a ROI
    x_margin = int(w * margin_ratio)
    y_margin = int(h * margin_ratio)


    # Verificar que l'etiqueta estigui dins dels marges de la imatge + marges afegits    
    x = max(0, x - x_margin)
    y = max(0, y - y_margin)
    x_end = min(x + w + 2 * x_margin, image.shape[1])
    y_end = min(y + h + 2 * y_margin, image.shape[0])

    region = image[y:y_end, x:x_end]

    blank_image = np.zeros((y_end - y, x_end - x, 3), dtype=np.uint8) # creació lienzo buit dimensió imatge original i 3 canals de colors RGB
    y_offset = max(0, blank_image.shape[0] - region.shape[0]) // 2
    x_offset = max(0, blank_image.shape[1] - region.shape[1]) // 2
    blank_image[y_offset:y_offset + region.shape[0], x_offset:x_offset + region.shape[1]] = region

    return blank_image, [[point[0] - x + x_offset, point[1] - y + y_offset] for point in shape['points']]

# funció creada per evitar apilaments en columnes/files i evitar que redimensioni lienzo sense aprofitament de la resat de pixels disponibles (corregit i funciona ok)
def create_square_mosaic(images, margin, original_height, original_width):
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))
    
    max_height = max(image.shape[0] for image in images) + margin
    max_width = max(image.shape[1] for image in images) + margin

    mosaic_height = max(original_height, grid_size * max_height - margin)
    mosaic_width = max(original_width, grid_size * max_width - margin)
    
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    positions = []
    
    current_x, current_y = margin, margin
    for idx, image in enumerate(images):
        if idx % grid_size == 0 and idx != 0:
            current_x = margin
            current_y += max_height

        if current_y + image.shape[0] > mosaic_height:
            mosaic_height += max_height + margin
            mosaic = np.pad(mosaic, ((0, max_height + margin), (0, 0), (0, 0)), 'constant')

        if current_x + image.shape[1] > mosaic_width:
            mosaic_width += max_width + margin
            mosaic = np.pad(mosaic, ((0, 0), (0, max_width + margin), (0, 0)), 'constant')

        positions.append((current_x, current_y))
        mosaic[current_y:current_y + image.shape[0], current_x:current_x + image.shape[1]] = image
        current_x += max_width

    return mosaic, positions

def augment_data(input_directory, output_directory, angles, scale_factors, shear_factors, class_list, margin=50):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    class_counts = {class_name: 0 for class_name in class_list}

    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            annotation_path = os.path.join(input_directory, filename)
            with open(annotation_path, 'r') as f:
                annotation_data = json.load(f)
            image_path = os.path.join(input_directory, annotation_data['imagePath'])
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                print(f"Warning: Unable to load image at path: {image_path}") # dir_2_augment manca imatge d'un json, sempre verificarà que exiseix la parella imatge-anotació
                continue
            
            original_height, original_width = original_image.shape[:2]
            
            for shape in annotation_data['shapes']:
                if shape['label'] in class_list:
                    augmented_images = []
                    new_shapes = []

                    # Original extraction per agafar el retall de l'anotació original
                    blank_image, new_points = extract_region(original_image, shape)
                    augmented_images.append(blank_image)
                    new_shapes.append({
                        'label': shape['label'],
                        'points': new_points,
                        'group_id': shape.get('group_id'),
                        'shape_type': shape['shape_type'],
                        'flags': shape.get('flags', {})
                    })
                    
                    # Rotations indicades llista angles
                    for angle in angles:
                        rotated_image, rotated_points = apply_rotation(original_image, shape['points'], angle)
                        blank_image, new_points = extract_region(rotated_image, {'points': rotated_points})
                        augmented_images.append(blank_image)
                        new_shapes.append({
                            'label': shape['label'],
                            'points': new_points,
                            'group_id': shape.get('group_id'),
                            'shape_type': shape['shape_type'],
                            'flags': shape.get('flags', {})
                        })

                    # Factor d'escala
                    for scale_factor in scale_factors:
                        scaled_image, scaled_points = apply_scale(original_image, shape['points'], scale_factor)
                        blank_image, new_points = extract_region(scaled_image, {'points': scaled_points})
                        augmented_images.append(blank_image)
                        new_shapes.append({
                            'label': shape['label'],
                            'points': new_points,
                            'group_id': shape.get('group_id'),
                            'shape_type': shape['shape_type'],
                            'flags': shape.get('flags', {})
                        })

                    # Shear per factors en ambdues coordenades del pla
                    for shear_factor in shear_factors:
                        for direction in ['x', 'y']:
                            sheared_image, sheared_points = apply_shear(original_image, shape['points'], shear_factor, direction)
                            blank_image, new_points = extract_region(sheared_image, {'points': sheared_points})
                            augmented_images.append(blank_image)
                            new_shapes.append({
                                'label': shape['label'],
                                'points': new_points,
                                'group_id': shape.get('group_id'),
                                'shape_type': shape['shape_type'],
                                'flags': shape.get('flags', {})
                            })

                    # Create and save mosaic
                    mosaic, positions = create_square_mosaic(augmented_images, margin, original_height, original_width)
                    class_index = class_counts[shape['label']]
                    mosaic_image_name = f"{os.path.basename(image_path).split('.')[0]}_{shape['label']}_{class_index}_mosaic.jpg"
                    mosaic_image_path = os.path.join(output_directory, mosaic_image_name)
                    cv2.imwrite(mosaic_image_path, mosaic)
                    
                    # Reubicar anotacions al mosaic (distancies evitar superposicio i ok)
                    for new_shape, (x_offset, y_offset) in zip(new_shapes, positions):
                        for point in new_shape['points']:
                            point[0] += x_offset
                            point[1] += y_offset
                    # anotacions corregides per evitar problemes LabelMe (corregit Vicenç i ok)
                    new_annotation_data = {
                        'version': annotation_data['version'],
                        'flags': annotation_data['flags'],
                        'shapes': new_shapes,
                        'imagePath': mosaic_image_name, # same as image so it can be read in LabelMe
                        'imageHeight': mosaic.shape[0],
                        'imageWidth': mosaic.shape[1],
                        'imageData': None  # Set imageData to null
                    }
                    mosaic_annotation_name = f"{os.path.basename(image_path).split('.')[0]}_{shape['label']}_{class_index}_mosaic.json"
                    mosaic_annotation_path = os.path.join(output_directory, mosaic_annotation_name)
                    with open(mosaic_annotation_path, 'w') as f:
                        json.dump(new_annotation_data, f, indent=4)
                    
                    class_counts[shape['label']] += 1

                    # per confirmar mateixa mida in i out mosaic retalls
                    # print(f"Original size: {original_width}x{original_height}, Mosaic size: {mosaic.shape[1]}x{mosaic.shape[0]}")

    print("Data augmentation process finished.") # per a confirmar que el programa ha acabat i check al LabelMe


# --- PARÀMETRES EXEMPLE ---
input_directory = '/Users/Xavier/Desktop/dataset_02/CB3p_second'
output_directory = '/Users/Xavier/Desktop/dataset_02/CB3p_mosaic'
angles = [20, -20]
scale_factors = [0.5, 1.5]
shear_factors = [0.2]
class_list = ['CB3p']

augment_data(input_directory, output_directory, angles, scale_factors, shear_factors, class_list)
