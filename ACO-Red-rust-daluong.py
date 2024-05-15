import numpy as np
import random
import copy
import cv2
import os
from numba import jit, types


class Ant:
    def __init__(self):
        self.path = []
        self.intensity_sum = 0.0


class AntColonySegmentation:
    def __init__(self, image, num_ants, max_iterations, alpha, beta, rho):
        self.image = image
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha  # pheromone importance
        self.beta = beta  # visibility (intensity) importance
        self.rho = rho  # pheromone evaporation rate
        self.green_found = True

        self.pheromones = np.ones([self.image.shape[0], self.image.shape[1]])

    @staticmethod
    def initialize_ants(num_ants, colony):
        ants = []
        for _ in range(num_ants):
            ant = Ant()
            ant.colony = colony
            ants.append(ant)
        return ants

    @staticmethod
    def clear_ant_paths(ants):
        for ant in ants:
            ant.path = []
            ant.intensity_sum = 0.0

    @staticmethod
    def construct_segment(image, ants):
        for ant in ants:
            current_position = (
                np.random.randint(0, image.shape[0]),
                np.random.randint(0, image.shape[1]),
            )
            ant.path.append(current_position)

            while True:
                neighbors = AntColonySegmentation.get_neighbors(image, current_position)
                probabilities = AntColonySegmentation.calculate_probabilities(
                    image, current_position, neighbors, ant
                )

                # Choose the next position based on probabilities
                next_position = np.random.choice(range(len(neighbors)), p=probabilities)
                next_position = neighbors[next_position]

                if (
                    image[next_position][2] <= image[next_position][1]
                    or image[next_position][2] <= image[next_position][0]
                ):
                    # If the pixel is not red, move to the next position
                    ant.path.append(next_position)
                    current_position = next_position
                else:
                    # If the pixel is red, stop moving
                    break

    @staticmethod
    def get_neighbors(image, position):
        height, width, _ = image.shape
        row, col = position
        neighbors = []
        neighbors_offsets = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),
        ]

        for offset_row, offset_col in neighbors_offsets:
            new_row, new_col = row + offset_row, col + offset_col
            if 0 <= new_row < image.shape[0] and 0 <= new_col < image.shape[1]:
                neighbors.append((new_row, new_col))

        return neighbors

    @staticmethod
    def calculate_probabilities(image, current_position, neighbors, ant):
        intensity = image[current_position][1]
        probabilities = []

        total_probability = 0  # Initialize total probability
        for neighbor in neighbors:
            attractiveness = AntColonySegmentation.calc_attractiveness(image, neighbor)
            pheromone = ant.intensity_sum
            probability = (pheromone**ant.colony.alpha) * (
                attractiveness**ant.colony.beta
            )
            if neighbor in ant.path:
                probability = 0
            probabilities.append(probability)
            total_probability += probability  # Sum up probabilities

        if total_probability == 0:
            # If all probabilities are 0, assign equal probabilities to all neighbors
            probabilities = [1 / len(neighbors)] * len(neighbors)
        else:
            # Normalize probabilities if total probability is greater than 0
            probabilities = [prob / total_probability for prob in probabilities]

        return probabilities

    @staticmethod
    def calc_attractiveness(image, position):
        row, col = position
        pixel = image[row, col]
        if (
            pixel[2] > pixel[1] and pixel[2] > pixel[0]
        ):  # Thay pixel[1] bằng pixel[2] để kiểm tra màu đỏ
            return 0.9  # Độ hấp dẫn cao đối với pixel màu đỏ
        else:
            return 0.1

    @staticmethod
    def update_pheromones(ants, pheromones):
        pheromones *= 1 - AntColonySegmentation.rho  # Evaporation
        for ant in ants:
            for position in ant.path:
                pheromones[position] += ant.intensity_sum

    def run(self):
        ants = AntColonySegmentation.initialize_ants(self.num_ants, self)
        best_ant = None
        segmentation_result = np.zeros_like(self.image)

        for iteration in range(self.max_iterations):
            AntColonySegmentation.clear_ant_paths(ants)
            AntColonySegmentation.construct_segment(self.image, ants)

            current_best_ant = max(ants, key=lambda ant: len(ant.path))
            if best_ant is None or len(current_best_ant.path) > len(best_ant.path):
                best_ant = copy.deepcopy(current_best_ant)

            # Fill color to the traversed path
            for position in current_best_ant.path:
                segmentation_result[position] = (
                    255,
                    255,
                    255,
                )  # White for traversed path

        # Fill color to the non-traversed area using the original image
        non_traversed_indices = np.where(segmentation_result == (0, 0, 0))
        segmentation_result[non_traversed_indices] = self.image[non_traversed_indices]

        return segmentation_result


num_ants = 100
max_iterations = 200
alpha = 0.9
beta = 0.9
rho = 0.1

# Khai báo đường dẫn thư mục chứa ảnh gốc và thư mục chứa ảnh sau khi ACO
input_folder = "Guava Dataset/Red_rust"
output_folder = "Processed Images"

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua các tệp trong thư mục đầu vào
for filename in os.listdir(input_folder):
    # Kiểm tra xem tệp
    if (
        filename.endswith(".jpg")
        or filename.endswith(".jpeg")
        or filename.endswith(".png")
    ):
        # Đoạn code xử lý ảnh
        image_path = os.path.join(input_folder, filename)

        # Đọc ảnh từ thư mục đầu vào
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # ACO cho ảnh
        aco_segmentation = AntColonySegmentation(
            image, num_ants, max_iterations, alpha, beta, rho
        )
        result = aco_segmentation.run()

        # Lưu ảnh đã xử lý vào thư mục đầu ra với cùng tên ảnh
        output_path = os.path.join(output_folder, filename)
        save = cv2.imwrite(output_path, result)
