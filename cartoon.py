import cv2
import numpy as np


def read_file(fileName):
    img = cv2.imread(fileName)
    return img


def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges


def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    ret, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result


def reduce_noise(img, diameter_pixel_neighborhood, sigma_color, sigma_space):
    blurred = cv2.bilateralFilter(
        img, d=diameter_pixel_neighborhood, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return blurred


def apply_edge_mask(blurred_image, edge_mask):
    cartoon = cv2.bitwise_and(blurred_image, blurred_image, edge_mask)
    return cartoon


if __name__ == "__main__":
    line_size = 5
    blur_size = 5
    total_color = 6

    img_source = read_file("IMG_1527.JPG")
    edges_output = edge_mask(img_source, line_size, blur_size)
    image_color = color_quantization(img_source, total_color)
    image_color_blurred = reduce_noise(image_color, 7, 200, 200)
    cartoon_result = apply_edge_mask(image_color_blurred, edges_output)

    cv2.imwrite("./output/test_edge.jpg", edges_output)
    cv2.imwrite("./output/test_color.jpg", image_color)
    cv2.imwrite("./output/test_color_less_noise.jpg",
                image_color_blurred)
    cv2.imwrite("./output/cartoon.jpg", cartoon_result)
