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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from PIL import Image
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
EPOCHS = 1
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


def build_model():
    base_model = MobileNet(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )

    model = models.Sequential(
        [
            base_model,
            layers.MaxPooling2D(),  # Add max pooling layer
            layers.Dense(2048, activation="relu"),
            layers.BatchNormalization(),  # Add batch normalization layer
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),  # Add batch normalization layer
            layers.Dropout(0.3),
            layers.Dense(512, activation="relu"),  # Add additional dense layer
            layers.BatchNormalization(),  # Add batch normalization layer
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),  # Add additional dense layer
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


# Chuyển đổi nhãn thành one-hot encoding
targets_one_hot = to_categorical(targets, num_classes)

checkpoint = ModelCheckpoint(
    "best_model_khongtinhchinhv3_v2_tangcuong.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)


class MetricsLogger(Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.header_written = False

    def on_epoch_end(self, epoch, logs=None):
        with open(self.log_file, "a") as f:
            if not self.header_written:
                f.write(
                    "Epoch\tTrain loss\tTrain accuracy\tval_loss\tval_accuracy\tval_recall\tval_precision\n"
                )
                self.header_written = True
            f.write(
                f"{epoch+1}\t{logs['loss']:.5f}\t{logs['accuracy']:.5f}\t{logs['val_loss']:.5f}\t{logs['val_accuracy']:.5f}\t{logs['val_recall']:.5f}\t{logs['val_precision']:.5f}\n"
            )


def save_confusion_matrix(y_true, y_pred, class_names, file_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(file_path, sep="\t")


def save_classification_report(y_true, y_pred, class_names, file_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(file_path, "a") as f:
        f.write(report)


for fold_no, (train_indices, test_indices) in enumerate(
    kfold.split(inputs, targets), 1
):
    # Reset model mỗi lần chạy fold mới
    model = build_model()
    # Khởi tạo MetricsLogger mới cho mỗi fold
    metrics_logger = MetricsLogger(
        f"metrics_khongtinhchinhv3_v2_tangcuong_fold_{fold_no}.log"
    )
    X_train, X_val, y_train, y_val = train_test_split(
        inputs, targets_one_hot, test_size=0.2, random_state=42
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

    save_confusion_matrix(
        targets[test_indices],
        y_pred,
        class_names,
        f"confusion_matrix_khongtinhchinhv3_v2_tangcuong.csv",
    )

    save_classification_report(
        targets[test_indices],
        y_pred,
        class_names,
        f"classification_report_khongtinhchinhv3_v2_tangcuong.txt",
    )
