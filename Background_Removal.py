import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

plt.rcParams["image.cmap"] = "gray"
plt.rcParams["figure.figsize"] = (16, 9)
# plt.style.use("dark_background")


def cieluv(img, target):
    # adapted from https://www.compuphase.com/cmetric.htm
    img = img.astype("int")
    aR, aG, aB = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    bR, bG, bB = target
    rmean = ((aR + bR) / 2.0).astype("int")
    r2 = np.square(aR - bR)
    g2 = np.square(aG - bG)
    b2 = np.square(aB - bB)

    # final sqrt removed for speed; please square your thresholds accordingly
    result = (((512 + rmean) * r2) >> 8) + 4 * g2 + (((767 - rmean) * b2) >> 8)

    return result


def process_image(f, plot=True):
    img = plt.imread(f)
    img_color = np.round(img * 255).astype("ubyte")[:, :, :3]  # Hình ảnh màu gốc
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)  # Chuyển đổi thành ảnh xám
    img_filter = (
        (cieluv(img_color, (71, 86, 38)) > 1600)
        & (cieluv(img_color, (65, 79, 19)) > 1600)
        & (cieluv(img_color, (95, 106, 56)) > 1600)
        & (cieluv(img_color, (56, 63, 43)) > 500)
    )
    img_color[img_filter] = 0

    img_median = cv2.medianBlur(img_color, 9)

    img_gray = cv2.cvtColor(img_median, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype("uint8")

    if plot:
        plt.figure(figsize=(16, 9))
        plt.subplot(141)
        plt.imshow(img)
        plt.title("Raw Image")

        plt.subplot(142)
        plt.imshow(img_color)
        plt.title("CIELUV Color Thresholding")

        plt.subplot(143)
        plt.imshow(img_median)
        plt.title("Median filter")

        plt.subplot(144)
        plt.imshow(img_gray, cmap="gray")
        plt.title("Black and White")
        plt.show()  # Hiển thị tất cả các subplot đã tạo ra trong quá trình xử lý hình ảnh
    return img_gray


i = process_image("Guava Dataset/Scab/Scab(1).jpg")
