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

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)

# Import ACO algorithm implementation library
from MetaheuristicOptimization import ACO


def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    return image_array


# Data directory
data_dir = "./Guava Dataset/"
class_names = os.listdir(data_dir)
num_classes = len(class_names)

# Load data from directory
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

# Convert labels to one-hot encoding
targets_one_hot = to_categorical(targets, num_classes)

# Define K-fold Cross Validation parameters
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1

# ACO parameters
aco_params = {
    "num_iterations": 100,
    "num_ants": 10,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.5,
}


def build_model():
    base_model = MobileNet(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )

    for layer in base_model.layers:
        layer.trainable = True

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


# Callback for logging metrics
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


# Callback for saving the best model
checkpoint = ModelCheckpoint(
    "best_model_MobileNet_ACO_CNN.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)


# Function to save confusion matrix
def save_confusion_matrix_append(y_true, y_pred, class_names, file_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    with open(file_path, "a") as f:
        df_cm.to_csv(f, sep="\t", mode="a")


# Function to save classification report
def save_classification_report(y_true, y_pred, class_names, file_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(file_path, "a") as f:
        f.write(report)


# Main training loop
for fold_no, (train_indices, test_indices) in enumerate(
    kfold.split(inputs, targets), 1
):
    X_train, X_val = inputs[train_indices], inputs[test_indices]
    y_train, y_val = targets_one_hot[train_indices], targets_one_hot[test_indices]

    # Reset model for each fold
    model = build_model()
    model.build((None, *IMG_SIZE, 3))
    model.summary()

    # Initialize MetricsLogger for each fold
    metrics_logger = MetricsLogger(
        f"metrics_MobileNet_ACO_CNN_fold_{fold_no}.log",
        X_val,
        y_val,
        fold_no,
        f"confusion_matrix_MobileNet_ACO_CNN",
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

    # ACO optimization for model training
    def fitness_func(solution):
        # Set model weights based on solution
        model.set_weights(solution)
        # Train the model
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=[checkpoint, metrics_logger],
            validation_data=(X_val, y_val),
        )
        # Calculate validation loss
        val_loss = history.history["val_loss"][-1]
        return val_loss

    # Initialize ACO optimizer
    aco = ACO(
        num_parameters=model.count_params(), fitness_function=fitness_func, **aco_params
    )

    # Optimize model parameters using ACO
    best_solution = aco.optimize()

    # Set model weights to the best solution found by ACO
    model.set_weights(best_solution)

    # Evaluate the model on test data
    scores = model.evaluate(
        inputs[test_indices], targets_one_hot[test_indices], verbose=1
    )
    print(
        f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
    )

    # Calculate metrics for test data
    y_pred = model.predict(inputs[test_indices])
    y_pred = np.argmax(y_pred, axis=1)

    # Save classification report
    save_classification_report(
        targets[test_indices],
        y_pred,
        class_names,
        f"classification_report_MobileNet_ACO_CNN.txt",
    )
