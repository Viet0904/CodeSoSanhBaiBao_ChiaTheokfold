import os
import numpy as np
import cv2
import random


class AntColonySegmentation:
    def __init__(self, image, num_ants, max_iterations, alpha, beta, rho):
        self.image = image
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha  # pheromone importance
        self.beta = beta  # visibility (intensity) importance
        self.rho = rho  # pheromone evaporation rate

        self.pheromones = np.ones([self.image.shape[0], self.image.shape[1]])

    def initialize_ants(self):
        ants = []
        for _ in range(self.num_ants):
            ant = {"path": [], "intensity_sum": 0.0}
            ants.append(ant)
        return ants

    def construct_segment(self, ant):
        current_position = (
            np.random.randint(0, self.image.shape[0]),
            np.random.randint(0, self.image.shape[1]),
        )
        ant["path"].append(current_position)

        for _ in range(10000):
            neighbors = self.get_neighbors(current_position)
            probabilities = self.calculate_probabilities(
                current_position, neighbors, ant["path"]
            )

            pom = [_ for _ in range(len(neighbors))]
            next_position = np.random.choice(pom, p=probabilities)
            next_position = neighbors[next_position]

            ant["path"].append(next_position)
            current_position = next_position

    def get_neighbors(self, position):
        height, width = self.image.shape  # Loại bỏ trích xuất số kênh màu
        row, col = position
        neighbors = []
        neighbors_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for offset_row, offset_col in neighbors_offsets:
            new_row, new_col = row + offset_row, col + offset_col
            if (
                0 <= new_row < height and 0 <= new_col < width
            ):  # Sử dụng height và width ở đây
                neighbors.append((new_row, new_col))

        return neighbors

    def calculate_probabilities(self, current_position, neighbors, path):
        intensity = self.image[current_position]
        probabilities = []

        for neighbor in neighbors:
            if neighbor not in path:
                neighbor_intensity = self.image[neighbor]
                pheromone = self.pheromones[neighbor]
                if neighbor_intensity > intensity:
                    visibility = 0.95
                else:
                    visibility = 0.05
                probability = (pheromone**self.alpha) * (visibility**self.beta)
                probabilities.append(probability)
            else:
                probabilities.append(0.0)

        probabilities = np.array(probabilities)
        if np.any(np.isnan(probabilities)):
            probabilities[np.isnan(probabilities)] = 0.1
        if np.any(np.isinf(probabilities)):
            probabilities[np.isinf(probabilities)] = 0.9
        total_probability = sum(probabilities)
        if total_probability > 0:
            probabilities /= total_probability
        else:
            probabilities = np.ones(len(neighbors)) / len(neighbors)

        return probabilities

    def update_pheromones(self, ants):
        self.pheromones *= 1 - self.rho
        for ant in ants:
            for position in ant["path"]:
                self.pheromones[position] += 1.0 / ant["intensity_sum"]

    def run(self):
        ants = self.initialize_ants()
        for iteration in range(self.max_iterations):

            for ant in ants:
                self.construct_segment(ant)
                ant["intensity_sum"] = sum(
                    self.image[position] for position in ant["path"]
                )

            self.update_pheromones(ants)

        best_ant = max(ants, key=lambda ant: ant["intensity_sum"])
        segmentation_result = np.zeros_like(self.image)

        for position in best_ant["path"]:
            segmentation_result[position] = self.image[position]

        return segmentation_result


# Đối số cho thuật toán ACO
num_ants = 5
max_iterations = 2
alpha = 1.0
beta = 2.0
rho = 0.5

# Đường dẫn đến thư mục chứa Guava Dataset
dataset_path = "Guava Dataset"

# Đường dẫn đến thư mục chứa nhãn "Red_rust"
label_path = os.path.join(dataset_path, "Scab")

# Kiểm tra xem thư mục tồn tại hay không
if os.path.isdir(label_path):
    # Lặp qua tất cả các tệp hình ảnh trong thư mục
    for image_file in os.listdir(label_path):
        # Kiểm tra xem tệp có phải là tệp hình ảnh hay không
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(label_path, image_file)
            # Kiểm tra xem tệp có tồn tại không
            if os.path.isfile(image_path):
                # Đọc ảnh
                image = cv2.imread(image_path)
                # Tiền xử lý ảnh: Áp dụng median filter với kernel kích thước 5x5
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_filtered = cv2.medianBlur(image_gray, 5)
                # Chạy thuật toán ACO trên ảnh đã được xử lý
                acs = AntColonySegmentation(
                    image_filtered, num_ants, max_iterations, alpha, beta, rho
                )
                segmented_image = acs.run()
                # Lưu trữ kết quả
                result_path = os.path.join("segmented_images", "Red_rust", image_file)
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                cv2.imwrite(result_path, segmented_image)
else:
    print("Thư mục nhãn 'Red_rust' không tồn tại.")
