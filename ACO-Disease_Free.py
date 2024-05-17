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

        while True:
            neighbors = self.get_neighbors(current_position)
            probabilities = self.calculate_probabilities(
                current_position, neighbors, ant
            )

            # Choose the next position based on probabilities
            next_position = np.random.choice(range(len(neighbors)), p=probabilities)
            next_position = neighbors[next_position]

            if (
                self.image[next_position][1] <= self.image[next_position][2]
                or self.image[next_position][1] <= self.image[next_position][0]
            ):

                # If the pixel is not green, move to the next position
                ant["path"].append(next_position)
                current_position = next_position
            else:
                # If the pixel is green, stop moving
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
            pixel[1] > pixel[0] and pixel[1] > pixel[2]
        ):  # Kiểm tra nếu pixel có màu xanh lá cây
            return 0.9  # Độ hấp dẫn cao đối với pixel màu xanh lá cây
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
                )  # Màu xanh lá cây cho đường đi

        # Fill color to the non-traversed area using the original image
        non_traversed_indices = np.where(segmentation_result == (0, 0, 0))
        segmentation_result[non_traversed_indices] = self.image[non_traversed_indices]

        return segmentation_result


# Đọc ảnh vào
image = cv2.imread("Guava Dataset/Disease_Free/Disease Free (2).jpg")

print(image.shape)

num_ants = 100
max_iterations = 200

alpha = 0.9
beta = 0.9
rho = 0.1

aco_segmentation = AntColonySegmentation(
    image, num_ants, max_iterations, alpha, beta, rho
)

result = aco_segmentation.run()
output_path = "Disease Free (2).jpg"
save = cv2.imwrite(output_path, result)
