import cv2
import numpy as np
import os


def preprocess(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    B, G, R = cv2.split(image)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    B = clahe.apply(B)
    G = clahe.apply(G)
    R = clahe.apply(R)

    clahe_image = cv2.merge([B, G, R])

    min_max_normalized_image = cv2.normalize(
        clahe_image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    return min_max_normalized_image


def generate_mask(image):
    gaussian_filtered = cv2.GaussianBlur(
        image,
        (25, 25),
        0
    )

    lab_image = cv2.cvtColor(
        gaussian_filtered,
        cv2.COLOR_BGR2Lab
    )

    l_mean = np.mean(lab_image[:, :, 0])
    a_mean = np.mean(lab_image[:, :, 1])
    b_mean = np.mean(lab_image[:, :, 2])

    _, l_binary = cv2.threshold(
        lab_image[:, :, 0],
        l_mean,
        255,
        cv2.THRESH_BINARY
    )

    _, a_binary = cv2.threshold(
        lab_image[:, :, 1],
        a_mean,
        255,
        cv2.THRESH_BINARY
    )

    _, b_binary = cv2.threshold(
        lab_image[:, :, 2],
        b_mean,
        255,
        cv2.THRESH_BINARY
    )

    intersection = cv2.bitwise_and(
        b_binary,
        cv2.bitwise_and(a_binary, l_binary)
    )

    median_filtered = cv2.medianBlur(
        intersection,
        15
    )

    return median_filtered


if __name__ == "__main__":
    print("Preprocessing module ready.")