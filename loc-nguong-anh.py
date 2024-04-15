import os
import cv2
import time


def main():
    # Bắt đầu tính thời gian chạy
    start = time.time()

    # Thiết lập các đường dẫn cố định
    input_dir = "./Guava Dataset"
    output_dir = "./output"
    nguong_duoi = 15
    nguong_tren = 255

    so_anh_da_xuly = 0

    # Phần xử lý, duyệt qua tất cả ảnh trong thư mục cần lọc ngưỡng
    for path, subdirs, files in os.walk(input_dir):
        for name in files:
            if name.endswith(".jpg"):
                # Đọc ảnh
                src = cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR)
                # Tạo ảnh với ngưỡng được thiết lập
                th, dst = cv2.threshold(
                    src, nguong_duoi, nguong_tren, cv2.THRESH_TOZERO
                )

                # Đường dẫn chứa ảnh đã tạo
                path_images = os.path.join(
                    output_dir + str(nguong_duoi) + "_" + str(nguong_tren),
                    path[len(input_dir) + 1 :],
                )
                # Nếu đường dẫn chưa tồn tại thì tạo mới
                if not os.path.exists(path_images):
                    os.makedirs(path_images)
                # Lưu ảnh vào đường dẫn đã tạo
                cv2.imwrite(os.path.join(path_images, name), dst)
                so_anh_da_xuly += 1

    # Kết thúc tính thời gian
    end = time.time()
    print(
        "Đã xử lý xong "
        + str(so_anh_da_xuly)
        + " files, thời gian chạy: "
        + str(end - start)
        + " giây "
    )


if __name__ == "__main__":
    main()
