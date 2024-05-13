import numpy as np
import cv2
import os
import random


class AntColonyColorSegmentation:
    def __init__(self, image, num_ants, max_iterations, alpha, beta, rho, sigma):
        self.image = image
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha  # pheromone importance
        self.beta = beta  # visibility (color similarity) importance
        self.rho = rho  # pheromone evaporation rate
        self.sigma = sigma  # parameter for color similarity calculation
        self.pheromones = np.ones_like(self.image)

    def initialize_ants(self):
        ants = []
        for _ in range(self.num_ants):
            ant = {"path": [], "color_sum": np.zeros_like(self.image)}
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

            next_position_index = np.random.choice(
                len(neighbors), p=np.ravel(probabilities) / np.sum(probabilities)
            ).item()

            next_position = neighbors[next_position_index]

            ant["path"].append(next_position)
            current_position = next_position

    def get_neighbors(self, position):
        height, width = self.image.shape[:2]
        row, col = position
        neighbors = []
        neighbors_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for offset_row, offset_col in neighbors_offsets:
            new_row, new_col = row + offset_row, col + offset_col
            if 0 <= new_row < height and 0 <= new_col < width:
                neighbors.append((new_row, new_col))

        return neighbors

    def calculate_probabilities(self, current_position, neighbors, path):
        color_current = self.image[current_position]
        probabilities = []
        for neighbor in neighbors:
            if neighbor not in path:
                color_neighbor = self.image[neighbor]
                pheromone = self.pheromones[neighbor]
                color_similarity = self.calculate_color_similarity(
                    color_current, color_neighbor
                )
                probability = (pheromone**self.alpha) * (color_similarity**self.beta)
                probabilities.append(probability)

        if not probabilities:  # Check if probabilities list is empty
            # Handle empty probabilities (e.g., choose random neighbor)
            return np.random.choice(neighbors)  # Return from within the function

        probabilities = np.array(probabilities)
        total_probability = np.sum(probabilities)
        if total_probability > 0:
            probabilities /= total_probability
        probabilities = np.ravel(probabilities)  # Convert to 1-dimensional array
        return probabilities

    def calculate_color_similarity(self, color1, color2):
        return np.exp(-np.linalg.norm(color1 - color2) / (2 * self.sigma**2))

    def update_pheromones(self, ants):
        self.pheromones *= 1 - self.rho
        for ant in ants:
            for position in ant["path"]:
                self.pheromones[position] += 1.0 / np.sum(ant["color_sum"])

    def run(self):
        ants = self.initialize_ants()
        for iteration in range(self.max_iterations):
            for ant in ants:
                self.construct_segment(ant)
                ant["color_sum"] = np.sum(
                    self.image[position] for position in ant["path"]
                )
            self.update_pheromones(ants)

        best_ant = max(ants, key=lambda ant: np.sum(ant["color_sum"]))
        segmentation_result = np.zeros_like(self.image)

        for position in best_ant["path"]:
            segmentation_result[position] = self.image[position]

        return segmentation_result


# Đối số cho thuật toán ACO


# Đường dẫn đến thư mục chứa Guava Dataset
dataset_path = "Guava Dataset"

# Đường dẫn đến thư mục chứa nhãn "Red_rust"
label_path = os.path.join(dataset_path, "Disease_Free")

# Kiểm tra xem thư mục tồn tại hay không
if os.path.isdir(label_path):
    # Chọn một ảnh trong thư mục
    image_file = "Guava Dataset/Disease_Free/Disease Free (1).jpg"  # Chọn ảnh cụ thể
    image_path = os.path.join(label_path, image_file)

    # Kiểm tra xem tệp có tồn tại không
    if os.path.isfile(image_path):
        # Đọc ảnh
        image = cv2.imread(image_path)
        # Tiền xử lý ảnh: Áp dụng median filter với kernel kích thước 5x5
        image_filtered = cv2.medianBlur(image, 5)
        # Đối số cho thuật toán ACO
        num_ants = 5
        max_iterations = 2
        alpha = 1.0
        beta = 2.0
        rho = 0.5
        sigma = 10.0  # Giá trị sigma cho tính toán tương đồng màu sắc
        # Chạy thuật toán ACO trên ảnh đã được xử lý
        acs = AntColonyColorSegmentation(
            image_filtered, num_ants, max_iterations, alpha, beta, rho, sigma
        )
        segmented_image = acs.run()
        # Lưu trữ kết quả
        result_path = os.path.join("segmented_images", "Red_rust", image_file)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, segmented_image)
else:
    print("Thư mục nhãn 'Red_rust' không tồn tại.")
