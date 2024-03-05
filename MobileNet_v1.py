import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import datetime
import os
from sklearn.model_selection import StratifiedKFold
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


class DetailedLoggingCallback(Callback):
    def __init__(self, test_data, file_prefix="MobileNet_v1_optAdam_lr0.001_bs32"):
        super(DetailedLoggingCallback, self).__init__()
        self.test_data = test_data
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.detail_file_path = f"{file_prefix}_{current_time}_details.txt"
        self.Confusion_Matrix_path = f"{file_prefix}_{current_time}_confusion_matrix"
        self.report_path = f"{file_prefix}_{current_time}_report"
        # Initialize file and write header with tab as separator
        with open(self.detail_file_path, "w") as f:
            f.write(
                "Epoch\tTrain Loss\tTrain Accuracy\tTest Loss\tTest Accuracy\tTest Precision\tTest Recall\tTest F1-Score\tTest MCC\tTest CMC\n"
            )
        with open(f"{self.Confusion_Matrix_path}_test.txt", "w") as f:
            f.write("Confusion Matrix Test\n")
        with open(f"{self.report_path}_test.txt", "w") as f:
            f.write("Classification Report Test\n")
        self.epoch_logs = []
        self.epoch_cm_logs = []
        self.epoch_report = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        test_results = self.model.evaluate(self.test_data, verbose=0)
        test_loss, test_accuracy = test_results[0], test_results[1]
        y_pred_test = np.argmax(self.model.predict(self.test_data), axis=1)
        y_true_test = self.test_data.classes
        cm_test = confusion_matrix(y_true_test, y_pred_test)
        report_test = classification_report(
            y_true_test, y_pred_test, digits=5, output_dict=True
        )
        precision_test = precision_score(y_true_test, y_pred_test, average="macro")
        recall_test = recall_score(y_true_test, y_pred_test, average="macro")
        f1_test = f1_score(y_true_test, y_pred_test, average="macro")
        mcc_test = matthews_corrcoef(y_true_test, y_pred_test)
        cmc_test = cohen_kappa_score(y_true_test, y_pred_test)
        cm_test = confusion_matrix(y_true_test, y_pred_test)
        report_test = classification_report(
            y_true_test, y_pred_test, digits=5, output_dict=True
        )
        print("Confusion Matrix (Test):")
        print(cm_test)
        print("Classification Report (Test):")
        print(report_test)
        print("Test Accuracy:", test_accuracy)
        print("Test Loss:", test_loss)
        print("Test Precision:", precision_test)
        print("Test Recall:", recall_test)
        print("Test F1-Score:", f1_test)
        print("Test MCC:", mcc_test)
        print("Test CMC:", cmc_test)
        self.epoch_cm_logs.append((epoch + 1, cm_test))
        self.epoch_report.append((epoch + 1, report_test))
        # Save information to temporary list with values separated by tab
        self.epoch_logs.append(
            (
                epoch + 1,
                logs.get("loss", 0),
                logs.get("accuracy", 0),
                test_loss,
                test_accuracy,
                precision_test,
                recall_test,
                f1_test,
                mcc_test,
                cmc_test,
            )
        )

    def on_train_end(self, logs=None):
        # Save information from each epoch to detail file, using tab as separator
        with open(self.detail_file_path, "a") as f:
            for log in self.epoch_logs:
                f.write(
                    f"{log[0]}\t{log[1]:.5f}\t{log[2]:.5f}\t{log[3]:.5f}\t{log[4]:.5f}\t{log[5]:.5f}\t{log[6]:.5f}\t{log[7]:.5f}\t{log[8]:.5f}\t{log[9]:.5f}\n"
                )
        with open(f"{self.Confusion_Matrix_path}_test.txt", "a") as f:
            for log in self.epoch_cm_logs:
                f.write(f"{log[1]}\n\n")
        with open(f"{self.report_path}_test.txt", "a") as f:
            for log in self.epoch_report:
                f.write(f"{log[1]}\n\n")


# Define paths to data directories
data_dir = "./Guava_Dataset"
sub_dirs = ["Disease_Free", "Phytopthora", "Red_rust", "Scab", "Styler_Root"]

# Create labels based on sub-directory names
labels = []
for idx, sub_dir in enumerate(sub_dirs):
    files = os.listdir(os.path.join(data_dir, sub_dir))
    labels.extend([idx] * len(files))

# Convert labels to numpy array
labels = np.array(labels)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 1


# Load the MobileNet model pre-trained weights
base_model = MobileNet(
    weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
)


# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False


# Initialize StratifiedKFold with 5 folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop through each fold
for fold_idx, (train_index, test_index) in enumerate(
    skf.split(np.zeros(len(labels)), labels)
):
    print(f"Fold {fold_idx + 1}")
    train_files = []
    test_files = []
    # Get file paths for train and test data based on fold indices
    for idx in train_index:
        train_files.append(os.path.join(data_dir, sub_dirs[labels[idx]], files[idx]))
    for idx in test_index:
        test_files.append(os.path.join(data_dir, sub_dirs[labels[idx]], files[idx]))

    # Data preprocessing and augmentation for train data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=False,
        fill_mode="nearest",
    )

    # Data preprocessing for test data
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load train data
    train_data = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    # Load test data
    test_data = test_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    # Initialize and compile model
    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    # Initialize callback for detailed logging
    detailed_logging_callback = DetailedLoggingCallback(test_data=test_data)

    # Train the model
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[detailed_logging_callback],
    )

    # Save the model
    model.save(f"./MobileNet_v1_fold{fold_idx + 1}.keras")
