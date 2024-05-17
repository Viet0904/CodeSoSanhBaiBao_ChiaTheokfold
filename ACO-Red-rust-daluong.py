import numpy as np
import random
import copy
import cv2
import os


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

    def initialize_ants(self):
        ants = []
        for _ in range(self.num_ants):
            ant = {"path": [], "intensity_sum": 0.0}
            ants.append(ant)
        return ants

    def clear_ant_paths(self, ants):
        for ant in ants:
            ant["path"] = []
            ant["intensity_sum"] = 0.0

    def construct_segment(self, ant):
        current_position = (
            np.random.randint(0, self.image.shape[0]),
            np.random.randint(0, self.image.shape[1]),
        )
        ant["path"].append(current_position)

        while True:
            neighbors = self.get_neighbors(current_position)
            probabilities = self.calculate_probabilities(
                current_position, neighbors, ant
            )

            # Choose the next position based on probabilities
            next_position = np.random.choice(range(len(neighbors)), p=probabilities)
            next_position = neighbors[next_position]

            if (
                self.image[next_position][2] <= self.image[next_position][1]
                or self.image[next_position][2] <= self.image[next_position][0]
            ):
                # If the pixel is not red, move to the next position
                ant["path"].append(next_position)
                current_position = next_position
            else:
                # If the pixel is red, stop moving
                break

    def get_neighbors(self, position):
        height, width, _ = self.image.shape
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
            if (
                0 <= new_row < self.image.shape[0]
                and 0 <= new_col < self.image.shape[1]
            ):
                neighbors.append((new_row, new_col))

        return neighbors

    def calculate_probabilities(self, current_position, neighbors, ant):
        intensity = self.image[current_position][1]
        probabilities = []

        total_probability = 0  # Initialize total probability
        for neighbor in neighbors:
            attractiveness = self.calc_attractiveness(neighbor)
            pheromone = self.pheromones[neighbor]
            probability = (pheromone**self.alpha) * (attractiveness**self.beta)
            if neighbor in ant["path"]:
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

    def calc_attractiveness(self, position):
        row, col = position
        pixel = self.image[row, col]
        if (
            pixel[2] > pixel[1] and pixel[2] > pixel[0]
        ):  # Thay pixel[1] bằng pixel[2] để kiểm tra màu đỏ
            return 0.9  # Độ hấp dẫn cao đối với pixel màu đỏ
        else:
            return 0.1

    def update_pheromones(self, ants):
        self.pheromones *= 1 - self.rho  # Evaporation
        for ant in ants:
            for position in ant["path"]:
                self.pheromones[position] += ant["intensity_sum"]

    def run(self):
        ants = self.initialize_ants()
        best_ant = None
        segmentation_result = np.zeros_like(self.image)

        for iteration in range(self.max_iterations):
            self.clear_ant_paths(ants)
            for ant in ants:
                self.construct_segment(ant)

            current_best_ant = max(ants, key=lambda ant: len(ant["path"]))
            if best_ant is None or len(current_best_ant["path"]) > len(
                best_ant["path"]
            ):
                best_ant = copy.deepcopy(current_best_ant)

            # Fill color to the traversed path
            for position in current_best_ant["path"]:
                segmentation_result[position] = (
                    255,
                    255,
                    255,
                )  # White for traversed path

        # Fill color to the non-traversed area using the original image
        non_traversed_indices = np.where(segmentation_result == (0, 0, 0))
        segmentation_result[non_traversed_indices] = self.image[non_traversed_indices]

        return segmentation_result


# Đọc ảnh vào
image = cv2.imread("Guava Dataset/Red_rust/Red Rust(87).jpg")
print(image.shape)
num_ants = 100
max_iterations = 150
alpha = 0.9
beta = 0.9
rho = 0.1
# Thư mục chứa ảnh đầu vào
input_folder = "Guava Dataset/Red_rust"

# Thư mục để lưu ảnh đã xử lý
output_folder = "Red_rust"
os.makedirs(output_folder, exist_ok=True)


# Lặp qua tất cả các tệp trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(
        ".png"
    ):  # Chỉ xử lý các tệp hình ảnh
        # Đọc hình ảnh
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        # Khởi tạo và chạy thuật toán phân đoạn ACO
        aco_segmentation = AntColonySegmentation(
            image, num_ants, max_iterations, alpha, beta, rho
        )
        result = aco_segmentation.run()

        # Tạo tên file đầu ra dựa trên tên file gốc
        output_filename = os.path.join(output_folder, filename)

        # Lưu kết quả phân đoạn vào thư mục đầu ra
        cv2.imwrite(output_filename, result)

        print("Đã xử lý và lưu ảnh", filename, "vào:", output_filename)
