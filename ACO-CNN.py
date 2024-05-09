import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold


def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)

    image = image.resize(target_size)

    image_array = np.array(image)

    image_array = image_array.astype("float32") / 255.0
    return image_array


# Thư mục chứa dữ liệu
data_dir = "./Guava Dataset/"

# List các tên lớp (tên thư mục trong data_dir)
class_names = os.listdir(data_dir)
num_classes = len(class_names)

# Load dữ liệu từ thư mục
inputs = []
targets = []

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 100

for class_index, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        inputs.append(preprocess_image(image_path))
        targets.append(class_index)

inputs = np.array(inputs)
targets = np.array(targets)


# Định nghĩa các tham số của K-fold Cross Validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
acc_per_fold = []
loss_per_fold = []

# Chuyển đổi nhãn thành one-hot encoding
targets_one_hot = to_categorical(targets, num_classes)

class MetricsLogger(Callback):
    def __init__(self, log_file, X_val, y_val, fold_no, log_file_prefix):
        super().__init__()
        self.log_file = log_file
        self.fold_no = fold_no
        self.log_file_prefix = log_file_prefix
        self.epoch_count = 0
        self.X_val = X_val
        self.y_val = y_val
        self.header_written = False

    def on_epoch_end(self, epoch, logs=None):
        with open(self.log_file, "a") as f:
            if not self.header_written:
                f.write(
                    "Epoch\tTrain loss\tTrain accuracy\tval_loss\tval_accuracy\tval_recall\tval_precision\tvalid_MCC\tvalid_CMC\tvalid_F1-Score\n"
                )
                self.header_written = True
            y_true = np.argmax(self.y_val, axis=1)
            y_pred = np.argmax(self.model.predict(self.X_val), axis=1)
            mcc = matthews_corrcoef(y_true, y_pred)
            cmc = cohen_kappa_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            f.write(
                f"{epoch+1}\t{logs['loss']:.5f}\t{logs['accuracy']:.5f}\t{logs['val_loss']:.5f}\t{logs['val_accuracy']:.5f}\t{logs['val_recall']:.5f}\t{logs['val_precision']:.5f}\t{mcc:.5f}\t{cmc:.5f}\t{f1:.5f}\n"
            )

        confusion_matrix_file = f"{self.log_file_prefix}_fold{self.fold_no}.txt"
        save_confusion_matrix_append(y_true, y_pred, class_names, confusion_matrix_file)

    def on_train_end(self, logs=None):
        print(f"Confusion matrix for fold {self.fold_no} has been saved.")


def save_confusion_matrix_append(y_true, y_pred, class_names, file_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    with open(file_path, "a") as f:
        df_cm.to_csv(f, sep="\t", mode="a")


def save_classification_report(y_true, y_pred, class_names, file_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(file_path, "a") as f:
        f.write(report)


def aco_image_segmentation(image):
    """
    Phân đoạn hình ảnh bằng thuật toán ACO.

    Args:
    image: Ma trận ảnh dạng (H, W, C), với H là chiều cao, W là chiều rộng và C là số kênh màu.
    Returns:
    segmentation: Ma trận ảnh dạng (H, W), với giá trị là nhãn phân đoạn cho mỗi pixel.
    """
    # Khởi tạo các tham số ACO
    n_ants = 10  # Số lượng kiến
    pheromone_decay = 0.5  # Hệ số suy giảm pheromone
    q = 0.9  # Hệ số cân bằng giữa pheromone và mật độ hình ảnh
    # Khởi tạo dấu vết pheromone
    pheromone = np.ones_like(image)
    # Khởi tạo kiến
    ants = [Ant(image.shape[0], image.shape[1]) for _ in range(n_ants)]
    # Lặp lại quá trình ACO
    for _ in range(100):
        # Di chuyển kiến
        for ant in ants:
            ant.move(image, pheromone, q)

    # Cập nhật dấu vết pheromone
    for ant in ants:
        pheromone = pheromone * (1 - pheromone_decay) + ant.pheromone_trail

    # Xác định nhãn phân đoạn cho mỗi pixel
    segmentation = np.zeros_like(image)
    for ant in ants:
        for i, j in ant.path:
            segmentation[i, j] = ant.id
    return segmentation


class Ant:
    """
    Lớp đại diện cho một con kiến trong thuật toán ACO.
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.position = (np.random.randint(0, height), np.random.randint(0, width))
        self.path = []
        self.pheromone_trail = np.zeros((height, width))

    def move(self, image, pheromone, q):
        """
        Di chuyển con kiến một bước.
        Args:
        image: Ma trận ảnh dạng (H, W, C), với H là chiều cao, W là chiều rộng và C là số kênh màu.
        pheromone: Ma trận ảnh dạng (H, W), với giá trị là nồng độ pheromone cho mỗi pixel.
        q: Hệ số cân bằng giữa pheromone và mật độ hình ảnh.
        """
        while True:
            new_position = self.get_next_position()
            # Tính toán xác suất di chuyển đến vị trí mới
            image_intensity = image[new_position]
            pheromone_intensity = pheromone[new_position]
            probability = (q * pheromone_intensity) / (
                (q * pheromone_intensity) + (1 - q) * image_intensity
            )
            # Kiểm tra xem có nên di chuyển đến vị trí mới hay không
            if np.random.random() < probability:
                break

            # Cập nhật vị trí hiện tại
            self.position = new_position

            # Thêm vị trí mới vào danh sách đường đi
            self.path.append(self.position)

            # Cập nhật dấu vết pheromone
            self.pheromone_trail[self.position] += 1

    def get_next_position(self):
        """
        Lấy vị trí tiếp theo của con kiến.

        Returns:
        new_position: Tuple (i, j), với i là tọa độ hàng và j là tọa độ cột của vị trí mới.
        """
        i, j = self.position
        # Lấy các vị trí lân cận
        neighbors = [
            (i + di, j + dj)
            for di in [-1, 0, 1]
            for dj in [-1, 0, 1]
            if 0 <= i + di < self.height
            and 0 <= j + dj < self.width
            and (di != 0 or dj != 0)
        ]
        # Loại bỏ các vị trí đã đi qua
        for neighbor in neighbors:
            if neighbor in self.path:
                neighbors.remove(neighbor)


def extract_features(image, segmentation):
    """
    Trích xuất đặc trưng từ hình ảnh phân đoạn bằng MobileNet.
    Args:
        image: Ma trận ảnh dạng (H, W, C), với H là chiều cao, W là chiều rộng và C là số kênh màu.
        segmentation: Ma trận ảnh dạng (H, W), với giá trị là nhãn phân đoạn cho mỗi pixel.
    Returns:
        features: Danh sách các ma trận đặc trưng, mỗi ma trận có dạng (n_regions, feature_vector_size).
    """
    # Khởi tạo MobileNet
    model = MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(image.shape[0], image.shape[1], 3),
    )
    # Trích xuất đặc trưng cho mỗi vùng phân đoạn
    features = []
    for label in np.unique(segmentation):
        region_mask = (segmentation == label).astype("float32")
        region_image = image * region_mask[:, :, np.newaxis]
        region_features = model.predict(region_image)
        features.append(region_features)
    return features


if __name__ == "__main__":
    # Đọc ảnh
    image = tf.keras.preprocessing.image.load_img("image.jpg", target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)

    # Phân đoạn hình ảnh bằng ACO
    segmentation = aco_image_segmentation(image)

    # Trích xuất đặc trưng bằng MobileNet
    features = extract_features(image, segmentation)

    # Sử dụng các đặc trưng cho các nhiệm vụ downstream
    # ...


def train_test_split_with_segmentation(inputs, targets):
    # Existing k-fold split logic
    # ...
    for fold_no, (train_indices, test_indices) in enumerate(
        kfold.split(inputs, targets), 1
    ):
        X_train, X_val = inputs[train_indices], inputs[test_indices]
        y_train, y_val = targets_one_hot[train_indices], targets_one_hot[test_indices]
        # Segment training images using ACO
        train_segmentations = aco_image_segmentation(inputs[train_indices])
        # Reset model mỗi lần chạy fold mới
        model = build_model()
        model.build((None, *IMG_SIZE, 3))
        model.summary()
        # Tính toán confusion matrix cho tập train trước khi tăng cường
        y_train_pred_before_augmentation = np.argmax(model.predict(X_train), axis=1)
        y_train_true = np.argmax(y_train, axis=1)
        confusion_matrix_train_before_augmentation = confusion_matrix(
            y_train_true, y_train_pred_before_augmentation
        )
        print("Confusion matrix for train data before augmentation:")
        print(confusion_matrix_train_before_augmentation)
        # Khởi tạo MetricsLogger mới cho mỗi fold
        metrics_logger = MetricsLogger(
            f"metrics_MobileNet_v3_v3_A_tangcuong_fold_{fold_no}.log",
            X_val,
            y_val,
            fold_no,
            f"confusion_matrix_MobileNet_v3_v3_A_tangcuong",
        )
        # Segment validation and test images (conceptual)
        val_segmentations = aco_image_segmentation(inputs[val_indices])  # Placeholder
        test_segmentations = aco_image_segmentation(inputs[test_indices])  # Placeholder

        # ... rest of the code within the loop (feature extraction, training, evaluation)

        # ... (use train_segmentations, val_segmentations, test_segmentations)
