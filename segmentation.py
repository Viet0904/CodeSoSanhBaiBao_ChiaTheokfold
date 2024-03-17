import os
import cv2
import numpy as np
from sklearn.model_selection import KFold
import random


# Function để đọc bounding box từ file label
def read_labels(label_file):
    with open(label_file, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.strip().split()
        labels.append(
            [
                int(line[0]),
                float(line[1]),
                float(line[2]),
                float(line[3]),
                float(line[4]),
            ]
        )
    return labels


# Function để vẽ bounding box và label
def draw_bounding_box(image, x, y, w, h, color):
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


# Function để thực hiện tăng cường dữ liệu trên tập train
def data_augmentation(image, labels):
    # Randomly flip horizontally
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        for label in labels:
            label[1] = 1 - label[1]  # Update x position for flipped image

    # Other augmentation techniques such as rotation, brightness adjustment, noise addition can be added here

    return image, labels


# Function để thực hiện instance segmentation và tăng cường dữ liệu trên tập train
def instance_segmentation(image_dir, label_dir, output_dir, augment=False):
    # Load pre-trained YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Load labels
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            label_file = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            if os.path.exists(label_file):
                labels = read_labels(label_file)

                # Perform data augmentation if needed
                if augment:
                    image, labels = data_augmentation(image, labels)

                # Process image, predict, draw bounding box, etc.
                height, width, _ = image.shape
                blob = cv2.dnn.blobFromImage(
                    image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
                )
                net.setInput(blob)
                outs = net.forward(output_layers)

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            for label in labels:
                                if label[0] == class_id:
                                    draw_bounding_box(
                                        image, x, y, w, h, (255, 0, 0)
                                    )  # Màu đỏ

                # Save augmented image and labels
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, image)

                with open(
                    os.path.join(
                        output_dir.replace("images", "labels"),
                        os.path.splitext(filename)[0] + ".txt",
                    ),
                    "w",
                ) as f:
                    for label in labels:
                        f.write(" ".join([str(x) for x in label]) + "\n")


# Thực hiện k-fold cross-validation
def k_fold_cross_validation(image_dir, label_dir, output_dir, k=5, augment=False):
    filenames = os.listdir(image_dir)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(filenames)):
        train_filenames = [filenames[idx] for idx in train_index]
        test_filenames = [filenames[idx] for idx in test_index]

        # Tạo thư mục cho từng fold
        fold_train_dir = os.path.join(output_dir, f"fold_{i+1}", "train")
        fold_test_dir = os.path.join(output_dir, f"fold_{i+1}", "test")
        os.makedirs(fold_train_dir, exist_ok=True)
        os.makedirs(fold_test_dir, exist_ok=True)

        # Sao chép các ảnh và label vào các thư mục tương ứng của mỗi fold
        for filename in train_filenames:
            image_src = os.path.join(image_dir, filename)
            label_src = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            image_dst = os.path.join(fold_train_dir, filename)
            label_dst = os.path.join(
                fold_train_dir.replace("images", "labels"),
                os.path.splitext(filename)[0] + ".txt",
            )
            os.makedirs(os.path.dirname(image_dst), exist_ok=True)
            os.makedirs(os.path.dirname(label_dst), exist_ok=True)
            os.system(f"cp {image_src} {image_dst}")
            os.system(f"cp {label_src} {label_dst}")

        for filename in test_filenames:
            image_src = os.path.join(image_dir, filename)
            label_src = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            image_dst = os.path.join(fold_test_dir, filename)
            label_dst = os.path.join(
                fold_test_dir.replace("test", "labels"),
                os.path.splitext(filename)[0] + ".txt",
            )
            os.makedirs(os.path.dirname(image_dst), exist_ok=True)
            os.makedirs(os.path.dirname(label_dst), exist_ok=True)
            os.system(f"cp {image_src} {image_dst}")
            os.system(f"cp {label_src} {label_dst}")

        # Thực hiện instance segmentation với tăng cường dữ liệu trên tập train
        instance_segmentation(
            fold_train_dir.replace("train", "images"),
            fold_train_dir.replace("images", "labels"),
            fold_train_dir,
            augment=augment,
        )
        instance_segmentation(
            fold_test_dir.replace("test", "images"),
            fold_test_dir.replace("images", "labels"),
            fold_test_dir,
        )


# Thực hiện k-fold cross-validation với tăng cường dữ liệu trên tập train
k_fold_cross_validation(
    "train/images", "train/labels", "k_fold_result_augmented", k=5, augment=True
)
