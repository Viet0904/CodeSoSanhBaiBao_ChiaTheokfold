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

    # def dead_end(self,neighbors):
    #     #neighbors = self.get_neighbors(pos)
    #     for n in neighbors:
    #         if self.image[n][1] > self.image[n][2] and self.image[n][1] > self.image[n][0]:
    #             return False
    #     return True

    def construct_segment(self, ant):
        current_position = (
            np.random.randint(0, self.image.shape[0]),
            np.random.randint(0, self.image.shape[1]),
        )
        ant["path"].append(current_position)

        briter = 0
        b = 0
        self.green_found = True
        while self.green_found:  # and briter<5000:
            briter += 1
            neighbors = self.get_neighbors(current_position)
            probabilities = self.calculate_probabilities(
                current_position, neighbors, ant
            )

            i = 2  # backtracking
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
                print("i==length")
                break

            if sum(probabilities) == 0:
                self.green_found = False
                break

            pom = [_ for _ in range(len(neighbors))]
            next_position = np.random.choice(pom, p=probabilities)
            next_position = neighbors[next_position]

            if (
                self.image[next_position][1] < self.image[next_position][2]
                or self.image[next_position][1] < self.image[next_position][0]
            ):
                b += 1
            else:
                b = 0

            if b == 50:
                self.green_found = False
                # print('green not found')
                break

            if next_position not in ant["path"]:
                ant["path"].append(next_position)
                current_position = next_position
        # print(briter)

    def get_neighbors(self, position):
        height, width, _ = self.image.shape
        row, col = position
        neighbors = []
        # neighbors_offsets = [(0, 1), (0, -1),(1,0),(-1,0)]
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

        # if self.dead_end(neighbors):
        #     probabilities = [0 for _ in range(len(neighbors))]
        #     return probabilit

        for neighbor in neighbors:
            attractiveness = self.calc_attractiveness(neighbor)
            # neighbor_intensity = self.image[neighbor][1]
            pheromone = self.pheromones[neighbor]
            probability = (pheromone**self.alpha) * (attractiveness**self.beta)
            # if probability == float('inf'):
            # probability = 10000.0
            if neighbor in ant["path"]:
                probability = 0
                # print([n in self.visited for n in neighbors])
            probabilities.append(probability)

        total_probability = sum(probabilities)
        # probabilities = [p / total_probability for p in probabilities]
        if total_probability > 0:
            probabilities /= total_probability

        return probabilities

    def calc_attractiveness(self, position):
        row, col = position
        pixel = self.image[row, col]
        if pixel[1] > pixel[2] and pixel[1] > pixel[2]:
            return 0.9  # High attractiveness for green pixels
        else:
            return 0.1

    def update_pheromones(self, ants):
        self.pheromones *= 1 - self.rho  # Evaporation
        for ant in ants:
            for position in ant["path"]:
                self.pheromones[position] += ant["intensity_sum"]  # sklonio 1.0/

    def run(self):
        ants = self.initialize_ants()
        best_ant = None
        segmentation_result = np.zeros_like(self.image)
        iter = 0

        for iteration in range(self.max_iterations):
            iter += 1
            self.clear_ant_paths(ants)
            # print('suma feromona',np.sum(self.pheromones))
            for ant in ants:
                self.construct_segment(ant)
                # ant['intensity_sum'] = sum(self.image[position][1] for position in ant['path'])
                ant["intensity_sum"] = sum(
                    (
                        self.image[position][1] > self.image[position][0]
                        and self.image[position][1] > self.image[position][2]
                    )
                    for position in ant["path"]
                )

            self.update_pheromones(ants)

            # Choose the best segment (ant) based on intensity sum
            current_best_ant = max(ants, key=lambda ant: ant["intensity_sum"])
            if (
                best_ant is None
                or current_best_ant["intensity_sum"] > best_ant["intensity_sum"]
            ):
                best_ant = copy.deepcopy(current_best_ant)

            print(iter, "len best path", len(current_best_ant["path"]))

            # segmentation_result = np.zeros_like(self.image)
            for position in current_best_ant["path"]:
                segmentation_result[position] = self.image[position]

        return segmentation_result


# Đối số cho thuật toán ACO
num_ants = 20
max_iterations = 20
alpha = 0.9
beta = 0.9
rho = 0.1

# Đường dẫn đến thư mục chứa Guava Dataset
dataset_path = "Guava Dataset"

# Đường dẫn đến thư mục chứa nhãn "Red_rust"
label_path = os.path.join(dataset_path, "Phytopthora")

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
                image_filtered = cv2.medianBlur(image, 5)
                # Chạy thuật toán ACO trên ảnh đã được xử lý
                acs = AntColonySegmentation(
                    image_filtered, num_ants, max_iterations, alpha, beta, rho
                )
                segmented_image = acs.run()
                # Lưu trữ kết quả
                result_path = os.path.join(
                    "segmented_images", "Phytopthora", image_file
                )
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                cv2.imwrite(result_path, segmented_image)
else:
    print("Thư mục nhãn 'Phytopthora' không tồn tại.")
