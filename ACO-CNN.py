import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    f1_score,
)
from PIL import Image
from sklearn.metrics import classification_report

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 100


def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)

    image = image.resize(target_size)

    image_array = np.array(image)

    image_array = image_array.astype("float32") / 255.0
    return image_array


class Ant:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.position = (np.random.randint(0, height), np.random.randint(0, width))
        self.path = []
        self.pheromone_trail = np.zeros((height, width))

    def move(self, image, pheromone, q):
        while True:
            new_position = self.get_next_position()
            image_intensity = image[new_position]
            pheromone_intensity = pheromone[new_position]
            probability = (q * pheromone_intensity) / (
                (q * pheromone_intensity) + (1 - q) * image_intensity
            )
            # Choose a random element from probability array
            random_index = np.random.randint(len(probability))
            if np.random.random() < probability[random_index]:
                break
            self.position = new_position
            self.path.append(self.position)
            self.pheromone_trail[self.position] += 1

    def get_next_position(self):
        i, j = self.position
        neighbors = [
            (i + di, j + dj)
            for di in [-1, 0, 1]
            for dj in [-1, 0, 1]
            if 0 <= i + di < self.height
            and 0 <= j + dj < self.width
            and (di != 0 or dj != 0)
        ]
        for neighbor in neighbors:
            if neighbor in self.path:
                neighbors.remove(neighbor)
        return neighbors[np.random.randint(len(neighbors))]


def aco_image_segmentation(image):
    n_ants = 10
    pheromone_decay = 0.5
    q = 0.9
    pheromone = np.ones_like(image)
    ants = [Ant(image.shape[0], image.shape[1]) for _ in range(n_ants)]
    for _ in range(100):
        for ant in ants:
            ant.move(image, pheromone, q)
    for ant in ants:
        ant_pheromone_trail_expanded = np.expand_dims(ant.pheromone_trail, axis=-1)
        pheromone = pheromone * (1 - pheromone_decay) + ant_pheromone_trail_expanded

    segmentation = np.zeros_like(image)
    for ant in ants:
        for i, j in ant.path:
            segmentation[i, j] = 1
    return segmentation


def extract_features(image, segmentation):
    model = MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(image.shape[0], image.shape[1], 3),
    )
    features = []
    for label in np.unique(segmentation):
        region_mask = (segmentation == label).astype("float32")
        region_image = image * region_mask[:, :, np.newaxis]
        region_features = model.predict(region_image)
        features.append(region_features)
    return features


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


# Define MobileNet architecture
base_model = MobileNet(
    weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
)

for layer in base_model.layers:
    layer.trainable = True


def build_model():
    model = models.Sequential(
        [
            base_model,
            layers.MaxPooling2D(),
            layers.Dense(2048, activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(5, activation="softmax"),
        ]
    )
    return model


def train_test_split_with_segmentation(inputs, targets, kfold, num_classes):
    num_folds = kfold.get_n_splits(inputs)
    for fold_no, (train_indices, test_indices) in enumerate(
        kfold.split(inputs, targets), 1
    ):
        X_train, X_test = inputs[train_indices], inputs[test_indices]
        y_train, y_test = targets[train_indices], targets[test_indices]
        train_segmentations = aco_image_segmentation(X_train)
        test_segmentations = aco_image_segmentation(X_test)
        y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
        y_test_one_hot = to_categorical(y_test, num_classes=num_classes)
        features_train = extract_features(X_train, train_segmentations)
        features_test = extract_features(X_test, test_segmentations)

        # Build model
        model = build_model()
        model.build((None, *IMG_SIZE, 3))
        model.summary()
        # Compile model
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        # Define data augmentation
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
        train_generator = train_datagen.flow(
            features_train, y_train_one_hot, batch_size=BATCH_SIZE
        )

        # Fit model with data augmentation
        history = model.fit(
            train_generator,
            verbose=1,
            epochs=100,
            validation_data=(features_test, y_test_one_hot),
            callbacks=[MetricsLogger(f"metrics_fold_{fold_no}.log", fold_no)],
        )


if __name__ == "__main__":
    # Load data
    data_dir = "./Guava Dataset/"
    class_names = os.listdir(data_dir)
    num_classes = len(class_names)
    inputs, targets = [], []
    IMG_SIZE = (224, 224)

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            inputs.append(preprocess_image(image_path))
            targets.append(class_index)

    inputs = np.array(inputs)
    targets = np.array(targets)

    # Define K-fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True)

    # Train-test split with segmentation
    train_test_split_with_segmentation(inputs, targets, kfold, num_classes)
