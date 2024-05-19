import numpy as np
import random
import copy
import cv2


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

        briter = 0
        b = 0
        self.green_found = True
        while self.green_found:
            briter += 1
            neighbors = self.get_neighbors(current_position)
            probabilities = self.calculate_probabilities(
                current_position, neighbors, ant
            )

            i = 2
            length = len(ant["path"])
            while sum(probabilities) == 0 and i <= length:
                current_position = ant["path"][-i]
                neighbors = self.get_neighbors(current_position)
                probabilities = self.calculate_probabilities(
                    current_position, neighbors, ant
                )
                i += 1

            if i == length and length > 2:
                self.green_found = False
                break

            if sum(probabilities) == 0:
                self.green_found = False
                break

            pom = [_ for _ in range(len(neighbors))]
            next_position = np.random.choice(pom, p=probabilities)
            next_position = neighbors[next_position]

            if (
                self.image[next_position][2] < self.image[next_position][1]
                or self.image[next_position][2] < self.image[next_position][0]
            ):  # Thay đổi điều kiện để kiểm tra màu đỏ
                b += 1
            else:
                b = 0

            if b == 50:
                self.green_found = False
                break

            if next_position not in ant["path"]:
                ant["path"].append(next_position)
                current_position = next_position

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

        for neighbor in neighbors:
            attractiveness = self.calc_attractiveness(neighbor)
            pheromone = self.pheromones[neighbor]
            probability = (pheromone**self.alpha) * (attractiveness**self.beta)
            if neighbor in ant["path"]:
                probability = 0
            probabilities.append(probability)

        total_probability = sum(probabilities)
        if total_probability > 0:
            probabilities /= total_probability

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
        iter = 0

        for iteration in range(self.max_iterations):
            iter += 1
            self.clear_ant_paths(ants)
            for ant in ants:
                self.construct_segment(ant)
                ant["intensity_sum"] = sum(
                    (
                        self.image[position][2] > self.image[position][1]
                        and self.image[position][2] > self.image[position][0]
                    )
                    for position in ant["path"]
                )

            self.update_pheromones(ants)

            current_best_ant = max(ants, key=lambda ant: ant["intensity_sum"])
            if (
                best_ant is None
                or current_best_ant["intensity_sum"] > best_ant["intensity_sum"]
            ):
                best_ant = copy.deepcopy(current_best_ant)

            print(iter, "len best path", len(current_best_ant["path"]))

            for position in current_best_ant["path"]:
                segmentation_result[position] = self.image[position]

        return segmentation_result


# Đọc ảnh vào
image = cv2.imread("Guava Dataset/Red_rust/Red Rust(87).jpg")

print(image.shape)

num_ants = 10
max_iterations = 10

alpha = 0.9
beta = 0.9
rho = 0.1

aco_segmentation = AntColonySegmentation(
    image, num_ants, max_iterations, alpha, beta, rho
)

result = aco_segmentation.run()
output_path = "Red Rust(87)_segmented.jpg"
save = cv2.imwrite(output_path, result)