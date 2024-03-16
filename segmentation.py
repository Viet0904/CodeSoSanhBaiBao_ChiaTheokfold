import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
)
from tensorflow.keras.optimizers import Adam

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = "D:/Guava Dataset"

# Danh sách các nhãn (tên thư mục con)
labels = os.listdir(data_dir)

# Đọc dữ liệu từ thư mục và tạo các cặp ảnh và nhãn tương ứng
images = []
masks = []
for label in labels:
    label_dir = os.path.join(data_dir, label)
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        mask_path = os.path.join(
            label_dir, img_name
        )  # Giả sử cùng tên file cho ảnh và nhãn
        image = load_img(img_path, target_size=(256, 256))
        mask = load_img(mask_path, target_size=(256, 256), color_mode="grayscale")
        images.append(img_to_array(image))
        masks.append(img_to_array(mask))

# Chuyển đổi list thành numpy array
images = np.array(images)
masks = np.array(masks)

# Chuẩn hóa dữ liệu ảnh và nhãn về khoảng [0, 1]
images = images / 255.0
masks = masks / 255.0

# Chuyển đổi nhãn thành dạng one-hot encoding
masks = to_categorical(masks, num_classes=len(labels))

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    images, masks, test_size=0.2, random_state=42
)


# Xây dựng mô hình Unet đơn giản
def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(pool2)
    concat1 = concatenate([conv1, up1], axis=3)
    conv3 = Conv2D(64, 3, activation="relu", padding="same")(concat1)
    conv4 = Conv2D(num_classes, 3, activation="softmax", padding="same")(conv3)

    model = Model(inputs=inputs, outputs=conv4)
    return model


# Khởi tạo và biên dịch mô hình
model = build_unet(input_shape=(256, 256, 3), num_classes=len(labels))
model.compile(
    optimizer=Adam(lr=1e-4), loss="categorical_crossentropy", metrics=["accuracy"]
)

# Huấn luyện mô hình
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Lưu mô hình
model.save("semantic_segmentation_model.h5")

# Dự đoán trên tập kiểm tra
predictions = model.predict(X_test)

# Lưu các ảnh dự đoán vào các thư mục tương ứng với nhãn của chúng
output_dir = "D:/Predictions"
for i, prediction in enumerate(predictions):
    # Lấy chỉ số của nhãn có xác suất cao nhất
    label_idx = np.argmax(prediction)
    label_name = labels[label_idx]

    # Tạo thư mục cho nhãn nếu chưa tồn tại
    label_dir = os.path.join(output_dir, label_name)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Lưu ảnh dự đoán vào thư mục tương ứng với nhãn
    pred_label = np.argmax(prediction, axis=-1)
    pred_label = np.expand_dims(pred_label, axis=-1) * (
        255 // (len(labels) - 1)
    )  # Chia tỷ lệ về khoảng [0, 255]
    pred_img = np.concatenate([pred_label] * 3, axis=-1).astype(np.uint8)
    img = Image.fromarray(pred_img.squeeze())
    img.save(os.path.join(label_dir, f"prediction_{i}.png"))

print("Đã lưu các ảnh dự đoán vào các thư mục tương ứng với nhãn của chúng.")
