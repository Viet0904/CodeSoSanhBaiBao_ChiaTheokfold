import cv2
import numpy as np


# Đọc và chuẩn bị ảnh
def preprocess_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    return image


# Khởi tạo pheromone và heuristic
def initialize_pheromone_and_heuristic(image):
    rows, cols, _ = image.shape
    pheromone = np.ones((rows, cols))
    heuristic = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            # Heuristic: giá trị cường độ màu (cường độ thấp -> có bệnh)
            heuristic[i][j] = np.mean(image[i, j]) / 255.0
    return pheromone, heuristic


# Cập nhật pheromone
def update_pheromone(pheromone, ants_paths, decay=0.1, Q=100):
    pheromone *= 1 - decay
    for path in ants_paths:
        for x, y in path:
            pheromone[x, y] += Q / len(path)
    return pheromone


# Thuật toán ACO
def ACO_segmentation(
    image, num_ants=10, num_iterations=10, alpha=1, beta=2, decay=0.1, Q=10
):
    pheromone, heuristic = initialize_pheromone_and_heuristic(image)
    rows, cols, _ = image.shape

    best_path = None
    best_cost = float("inf")

    for iteration in range(num_iterations):
        ants_paths = []

        for ant in range(num_ants):
            path = []
            x, y = np.random.randint(rows), np.random.randint(cols)

            while not is_infected(image[x, y]):
                path.append((x, y))
                next_x, next_y = select_next_pixel(
                    pheromone, heuristic, x, y, alpha, beta
                )
                x, y = next_x, next_y

            ants_paths.append(path)
            if len(path) < best_cost:
                best_cost = len(path)
                best_path = path

        pheromone = update_pheromone(pheromone, ants_paths, decay, Q)

    segmented_image = np.zeros((rows, cols), dtype=np.uint8)
    for x, y in best_path:
        segmented_image[x, y] = 255

    return segmented_image


# Hàm kiểm tra nếu pixel bị nhiễm bệnh
def is_infected(pixel):
    return np.mean(pixel) < 128


# Hàm chọn pixel tiếp theo
def select_next_pixel(pheromone, heuristic, x, y, alpha, beta):
    neighbors = get_neighbors(x, y, pheromone.shape)
    probabilities = []

    for nx, ny in neighbors:
        tau = pheromone[nx, ny] ** alpha
        eta = heuristic[nx, ny] ** beta
        probabilities.append(tau * eta)

    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    next_index = np.random.choice(len(neighbors), p=probabilities)
    return neighbors[next_index]


# Hàm lấy hàng xóm của pixel hiện tại
def get_neighbors(x, y, shape):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                neighbors.append((nx, ny))
    return neighbors


# Đọc ảnh và thực hiện phân đoạn
image_path = "Guava Dataset/Red_rust/Red Rust(87).jpg"
image = preprocess_image(image_path)
segmented_image = ACO_segmentation(image)

output_path = "output.jpg"
cv2.imwrite(output_path, segmented_image)
print("Segmented image saved at", output_path)
