import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import datetime
import pandas as pd
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)


def preprocess_image_aco(image_path):
    # Đọc ảnh từ đường dẫn
    image = Image.open(image_path)

    # Resize ảnh đến kích thước mong muốn
    image = image.resize(IMG_SIZE)

    # Chuyển đổi ảnh sang numpy array và chuẩn hóa giá trị pixel về khoảng [0, 1]
    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0

    # Trả về ảnh đã được tiền xử lý
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
        inputs.append(preprocess_image_aco(image_path))  # Sử dụng preprocess_image_aco
        targets.append(class_index)


inputs = np.array(inputs)
targets = np.array(targets)


# Định nghĩa các tham số của K-fold Cross Validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
acc_per_fold = []
loss_per_fold = []


def build_model(input_shape):
    base_model = MobileNet(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    for layer in base_model.layers:
        layer.trainable = True

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(name="val_precision"),
            tf.keras.metrics.Recall(name="val_recall"),
        ],
    )

    return model


def extract_features(image):
    # Khởi tạo ma trận pheromone và visibility matrix
    pheromone_matrix = np.ones((image.shape[1], image.shape[2])) * 0.01
    visibility_matrix = np.ones((image.shape[1], image.shape[2]))
    ant_count = 10
    iteration_count = 10
    # Thực hiện quá trình di chuyển của kiến
    for iteration in range(iteration_count):
        for ant in range(ant_count):
            # Khởi tạo vị trí bắt đầu của kiến
            current_position = [
                np.random.randint(0, image.shape[1]),
                np.random.randint(0, image.shape[2]),
            ]
            # Thực hiện di chuyển của kiến
            for step in range(image.shape[1] * image.shape[2]):
                # Tính toán xác suất di chuyển
                transition_probabilities = (
                    pheromone_matrix[current_position[0], current_position[1]]
                    * visibility_matrix[current_position[0], current_position[1]]
                )
                transition_probabilities = transition_probabilities / np.sum(
                    transition_probabilities
                )
                # Lựa chọn vị trí tiếp theo
                next_position = np.unravel_index(
                    np.random.choice(
                        np.arange(image.shape[1] * image.shape[2]),
                        p=transition_probabilities.ravel(),
                    ),
                    (image.shape[1], image.shape[2]),
                )
                # Cập nhật pheromone
                pheromone_matrix[next_position[0], next_position[1]] += 1.0
                # Cập nhật vị trí hiện tại
                current_position = next_position
    # Trả về vùng bị nhiễm bệnh
    return pheromone_matrix


# Chuyển đổi nhãn thành one-hot encoding
targets_one_hot = to_categorical(targets, num_classes)

checkpoint = ModelCheckpoint(
    "best_model_MobileNetC_tangcuongv1.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)


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


for fold_no, (train_indices, test_indices) in enumerate(
    kfold.split(inputs, targets), 1
):
    # Reset model mỗi lần chạy fold mới
    # Sửa lại hàm build_model để chấp nhận kích thước đầu vào mới từ ACO
    input_shape_aco = extracted_features.shape[
        1:
    ]  # Định dạng kích thước đầu vào từ ACO
    model = build_model(input_shape_aco)
    model.build((None, *IMG_SIZE, 3))
    model.summary()
    # Thực hiện thuật toán ACO để trích xuất đặc điểm từ ảnh
    extracted_features = []
    for image_path in list_of_image_paths:
        image = preprocess_image_aco(image_path)  # Tiền xử lý ảnh
        features = extract_features_aco(
            image
        )  # Trích xuất đặc điểm bằng thuật toán ACO
        extracted_features.append(features)

    extracted_features = np.array(extracted_features)
    X_train, X_val, y_train, y_val = train_test_split(
        extracted_features, targets_one_hot, test_size=0.2, random_state=42
    )

    # Khởi tạo MetricsLogger mới cho mỗi fold
    metrics_logger = MetricsLogger(
        f"metrics_MobileNetC_tangcuongv1_fold_{fold_no}.log",
        X_val,
        y_val,
        fold_no,
        f"confusion_matrix_MobileNetC_tangcuongv1",
    )

    # Khởi tạo ImageDataGenerator để áp dụng tăng cường dữ liệu cho tập huấn luyện của fold hiện tại
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=False,
        fill_mode="nearest",
    )
    # Tạo ra dữ liệu augmented từ dữ liệu train
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

    # Huấn luyện mô hình với dữ liệu tăng cường của fold hiện tại
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[checkpoint, metrics_logger],
        validation_data=(X_val, y_val),
    )
    # Đánh giá mô hình trên dữ liệu kiểm tra của fold hiện tại
    scores = model.evaluate(
        inputs[test_indices], targets_one_hot[test_indices], verbose=1
    )
    print(
        f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
    )

    # Tính toán các metric
    y_pred = model.predict(inputs[test_indices])
    y_pred = np.argmax(y_pred, axis=1)

    save_classification_report(
        targets[test_indices],
        y_pred,
        class_names,
        f"classification_report_MobileNetC_tangcuongv1.txt",
    )
