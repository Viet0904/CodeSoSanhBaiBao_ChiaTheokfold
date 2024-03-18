import os
import cv2
import numpy as np
from PIL import Image

# Định nghĩa biến
data_path = "./Guava Dataset"


# Hàm tạo mask cho từng ảnh
def create_mask(image_path, mask_folder):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)
    # Chuyển đổi ảnh sang ảnh grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thực hiện xử lý ảnh để tạo mask, ví dụ: phát hiện cạnh
    _, mask = cv2.threshold(grayscale_image, 100, 255, cv2.THRESH_BINARY)
    # Chuyển mask thành ảnh PIL
    mask_image = Image.fromarray(mask)
    # Lưu mask vào thư mục mask
    mask_image.save(os.path.join(mask_folder, os.path.basename(image_path)))


# Hàm tạo mask cho từng lớp
def make_masks_for_classes(data_path):
    # Lặp qua từng lớp
    for folder in os.listdir(data_path):
        if not folder.startswith("."):  # Bỏ qua các thư mục ẩn (nếu có)
            print("*" * 10, folder)

            # Tạo đường dẫn tới thư mục lớp và tạo thư mục mask tương ứng
            class_folder = os.path.join(data_path, folder)
            mask_folder = os.path.join(data_path, folder + "_mask")
            if not os.path.exists(mask_folder):
                os.mkdir(mask_folder)

            # Nếu thư mục lớp chứa các file ảnh
            if os.path.isdir(class_folder):
                # Thực hiện tạo mask cho từng ảnh trong thư mục lớp
                for file in os.listdir(class_folder):
                    if file.endswith(".jpg"):  # Xác định các file ảnh JPG
                        # Đường dẫn đến ảnh
                        image_path = os.path.join(class_folder, file)
                        # Tạo mask cho ảnh và lưu vào thư mục mask
                        create_mask(image_path, mask_folder)


# Gọi hàm để tạo mask cho các lớp trong dataset
make_masks_for_classes(data_path)
