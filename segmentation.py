import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    concatenate,
    Conv2DTranspose,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


def load_data(data_path, target_size=(256, 256)):
    images = []
    masks = []
    for class_name in os.listdir(data_path):
        if class_name.endswith("_mask"):
            continue
        class_path = os.path.join(data_path, class_name)
        mask_path = os.path.join(data_path, f"{class_name}_mask")
        for filename in os.listdir(class_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(class_path, filename)
                mask_image_filename = filename.split(".")[0] + ".jpg"
                mask_image_path = os.path.join(mask_path, mask_image_filename)
                if os.path.isfile(image_path) and os.path.isfile(mask_image_path):
                    image = cv2.imread(image_path)
                    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

                    # Chuyển đổi kích thước ảnh và mask
                    image = cv2.resize(image, target_size)
                    mask = cv2.resize(mask, target_size)

                    images.append(image)
                    masks.append(mask)
    return np.array(images), np.array(masks)


# Hàm xây dựng mô hình U-Net
def unet(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation="relu", padding="same")(
        Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(drop5)
    )
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same")(
        Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv6)
    )
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same")(
        Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv7)
    )
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same")(
        Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv8)
    )
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)
    conv9 = Conv2D(2, 3, activation="relu", padding="same")(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


# Thư mục chứa dữ liệu
data_path = "./Guava Dataset"

# Load dữ liệu
images, masks = load_data(data_path)

# Chia dữ liệu thành tập train và tập validation
train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu
train_images = train_images / 255.0
val_images = val_images / 255.0
train_masks = train_masks / 255.0
val_masks = val_masks / 255.0

# Xây dựng mô hình
input_shape = (256, 256, 3)  # Kích thước ảnh đầu vào
model = unet(input_shape)

# Compile mô hình
model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Định nghĩa callback để lưu mô hình tốt nhất
checkpoint = ModelCheckpoint(
    "unet_model.h5", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)

# Huấn luyện mô hình
history = model.fit(
    train_images,
    train_masks,
    validation_data=(val_images, val_masks),
    batch_size=16,
    epochs=50,
    callbacks=[checkpoint],
)
