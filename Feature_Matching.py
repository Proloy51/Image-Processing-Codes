#Feature Matching
#1907016



import numpy as np
import cv2
from tabulate import tabulate


def find_boundary(binary_image):
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    border_image = cv2.absdiff(binary_image, eroded_image)

    return border_image


def calculate_parameters(binary_image, border_image):
    area = np.count_nonzero(binary_image)
    perimeter = np.count_nonzero(border_image)
    coords = np.column_stack(np.where(binary_image > 0)).astype(float)
    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)
    max_diameter = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

    return area, perimeter, max_diameter


def calculate_descriptors(area, perimeter, max_diameter):
    form_factor = 4 * np.pi * area / (perimeter ** 2)
    roundness = (4 * area) / (np.pi * max_diameter ** 2)
    compactness = (perimeter ** 2) / area

    return form_factor, roundness, compactness


def cosser(descriptor1, descriptor2):
    dot_product = 0
    for i in range(len(descriptor1)):
        dot_product += descriptor1[i] * descriptor2[i]

    norm1 = 0
    for a in descriptor1:
        norm1 += a ** 2
    norm1 = np.sqrt(norm1)

    norm2 = 0
    for b in descriptor2:
        norm2 += b ** 2
    norm2 = np.sqrt(norm2)

    return dot_product / (norm1 * norm2)


def match_descriptors(train_descriptors, test_descriptors):
    similarity_matrix = np.zeros((len(test_descriptors), len(train_descriptors)))

    for i, test_desc in enumerate(test_descriptors):
        for j, train_desc in enumerate(train_descriptors):
            similarity_matrix[i, j] = cosser(test_desc, train_desc)

    return similarity_matrix


def shobcalc(image_path):
    binary_image = cv2.imread(image_path, 0)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    border_image = find_boundary(binary_image)
    area, perimeter, max_diameter = calculate_parameters(binary_image, border_image)
    descriptors = calculate_descriptors(area, perimeter, max_diameter)
    return descriptors


def buildsim(similarity_matrix, train_images, test_images):
    table = [["Train/Test"] + train_images]
    for i, test_image in enumerate(test_images):
        row = [test_image] + [f"{similarity_matrix[i, j]:.4f}" for j in range(len(train_images))]
        table.append(row)
    return table


def simwrite(file_path, simtable):
    with open(file_path, 'w') as file:
        file.write(tabulate(simtable, headers="firstrow", tablefmt="grid"))

train_images = ['c1.jpg', 'p1.png', 't1.jpg']
test_images = ['c2.jpg', 'p2.png', 'p3.jpg', 'st.jpg', 't2.jpg']

train_descriptors = [shobcalc(img) for img in train_images]
test_descriptors = [shobcalc(img) for img in test_images]

similarity_matrix = match_descriptors(train_descriptors, test_descriptors)
similarity_table = buildsim(similarity_matrix, train_images, test_images)

output_file_path = 'output.txt'
simwrite(output_file_path, similarity_table)
print(f"file shesh {output_file_path}")




