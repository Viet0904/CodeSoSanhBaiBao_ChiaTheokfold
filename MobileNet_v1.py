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
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold

from PIL import Image
import numpy as np


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess the image from the given path.

    Parameters:
        image_path (str): The path to the image file.
        target_size (tuple): The target size to resize the image to.

    Returns:
        numpy.ndarray: The preprocessed image as a numpy array.
    """
    # Load the image
    image = Image.open(image_path)
    # Resize the image to the target size
    image = image.resize(target_size)
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the pixel values to the range [0, 1]
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
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 100
for class_index, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        # Đọc ảnh và tiền xử lý (ví dụ: resize, chuyển đổi sang numpy array)
        # Đảm bảo rằng inputs và targets tương thích với mô hình của bạn
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

base_model = MobileNet(
    weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
)


# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False
# Định nghĩa cấu trúc mô hình
model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

# Chuyển đổi nhãn thành one-hot encoding
targets_one_hot = to_categorical(targets, num_classes)

# Compile mô hình
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

# Tạo list để lưu thông tin từ các epoch
epoch_list = []
train_loss_list = []
train_accuracy_list = []
test_loss_list = []
test_accuracy_list = []
test_precision_list = []
test_recall_list = []
test_f1_score_list = []
test_mcc_list = []
test_cmc_list = []

# Định nghĩa callback để lưu model có độ chính xác cao nhất trên tập kiểm tra
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)
# Lặp qua các fold của K-fold Cross Validation
for train_indices, test_indices in kfold.split(inputs, targets):
    # Huấn luyện mô hình trên dữ liệu của fold hiện tại
    history = model.fit(
        inputs[train_indices],
        targets_one_hot[train_indices],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[checkpoint],
        validation_data=(inputs[test_indices], targets_one_hot[test_indices]),
    )

    # Đánh giá mô hình trên dữ liệu kiểm tra của fold hiện tại
    scores = model.evaluate(
        inputs[test_indices], targets_one_hot[test_indices], verbose=1
    )
    print(
        f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
    )
    # Lưu thông tin từ các epoch vào list
    epoch_list.append(fold_no)
    train_loss_list.append(history.history["loss"][-1])
    train_accuracy_list.append(history.history["accuracy"][-1])
    test_loss_list.append(scores[0])
    test_accuracy_list.append(scores[1])

    # Tính toán các metric khác
    y_pred = model.predict(inputs[test_indices])
    y_pred = np.argmax(y_pred, axis=1)
    precision = precision_score(targets[test_indices], y_pred, average="macro")
    recall = recall_score(targets[test_indices], y_pred, average="macro")
    f1_score_val = f1_score(targets[test_indices], y_pred, average="macro")
    # Initialize an empty list to store the MCC values for each class
    mcc_per_class = []

    # Calculate MCC for each class
    for class_index, class_name in enumerate(class_names):
        mcc = matthews_corrcoef(
            targets[test_indices] == class_index, y_pred == class_index
        )
        mcc_per_class.append(mcc)
    cmc = cohen_kappa_score(targets[test_indices], y_pred)

    # Lưu các metric vào list
    test_precision_list.append(precision)
    test_recall_list.append(recall)
    test_f1_score_list.append(f1_score_val)
    test_mcc_list.append(mcc)
    test_cmc_list.append(cmc)

    # Lưu confusion matrix vào file
    cm = confusion_matrix(targets[test_indices], y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"confusion_matrix_fold.csv", sep="\t")

    # Lưu classification report vào file
    cr = classification_report(targets[test_indices], y_pred)
    with open(f"classification_report_fold.txt", "w") as f:
        f.write(cr)

    # Tăng số fold
    fold_no += 1


# Tạo DataFrame từ thông tin đã lưu
log_data = {
    "Epoch": epoch_list,
    "Train Loss": train_loss_list,
    "Train Accuracy": train_accuracy_list,
    "Test Loss": test_loss_list,
    "Test Accuracy": test_accuracy_list,
    "Test Precision": test_precision_list,
    "Test Recall": test_recall_list,
    "Test F1-Score": test_f1_score_list,
    "Test MCC": test_mcc_list,
    "Test CMC": test_cmc_list,
}
log_df = pd.DataFrame(log_data)

# Lưu DataFrame thành file CSV
log_df.to_csv("log_file.csv", index=False)
