import cv2
from PIL import Image
import numpy as np


# This function applies to images mean filter.
def mean_filtering(img, ws):

    for i in range(ws//2, img.shape[0] - ws//2):
        for j in range(ws//2, img.shape[1] - ws//2):
            for c in range(img.shape[2]):
                img[i][j][c] = np.mean(img[i - ws//2: i + ws//2 + 1, j - ws//2: j + ws//2 + 1, c])

    return Image.fromarray(img.astype(np.uint8))


# This function applies to image gaussian filter.
def gaussian_filtering(img, ws, sigma):

    x = np.arange(-ws//2 + 1, ws//2 + 1)  # x = [0, 1, 2]    if window size is 3x3
    y = np.arange(-ws//2 + 1, ws//2 + 1)  # y = [0, 1, 2]    if window size is 3x3
    x, y = np.meshgrid(x, y, sparse=True)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2)))  # G(x,y)
    gaussian_filter = g / g.sum()

    for i in range(ws//2, img.shape[0] - ws//2):
        for j in range(ws//2, img.shape[1] - ws//2):
            for c in range(img.shape[2]):
                img[i][j][c] = np.sum(np.multiply((img[i - ws//2: i + ws//2 + 1, j - ws//2: j + ws//2 + 1, c]), gaussian_filter))  # CORRELATION

    return Image.fromarray(img.astype(np.uint8))


# This function applies to images kuwahara filter.
def kuwahara_filtering(img, ws):

    rgb = np.array(img, dtype=float)
    hsv = np.array(img.convert("HSV"), dtype=float)
    h, s, v = cv2.split(hsv)
    quadrant_size = ws//2

    for y in range(quadrant_size, hsv.shape[0] - quadrant_size):
        for x in range(quadrant_size, hsv.shape[1] - quadrant_size):
            v_channel = v[y - ws//2: y + ws//2 + 1, x - ws//2: x + ws//2 + 1]

            # Q1, Q2, Q3, Q4 blocks
            Q1 = v_channel[0: v_channel.shape[1] // 2 + 1, v_channel.shape[0] // 2: v_channel.shape[0]]
            Q2 = v_channel[0: v_channel.shape[1] // 2 + 1, 0: v_channel.shape[0] // 2 + 1]
            Q3 = v_channel[v_channel.shape[1] // 2: v_channel.shape[1], 0: v_channel.shape[0] // 2 + 1]
            Q4 = v_channel[v_channel.shape[1] // 2: v_channel.shape[1], v_channel.shape[0] // 2: v_channel.shape[0]]

            # Standard Deviations
            std_Q1 = np.std(Q1)
            std_Q2 = np.std(Q2)
            std_Q3 = np.std(Q3)
            std_Q4 = np.std(Q4)

            st_deviations = np.array([std_Q1, std_Q2, std_Q3, std_Q4])
            min = st_deviations.argmin()

            if min == 0:
                rgb[y][x][0] = np.mean(rgb[y - quadrant_size: y + 1, x: x + quadrant_size + 1, 0])
                rgb[y][x][1] = np.mean(rgb[y - quadrant_size: y + 1, x: x + quadrant_size + 1, 1])
                rgb[y][x][2] = np.mean(rgb[y - quadrant_size: y + 1, x: x + quadrant_size + 1, 2])
            if min == 1:
                rgb[y][x][0] = np.mean(rgb[y - quadrant_size: y + 1, x - quadrant_size: x + 1, 0])
                rgb[y][x][1] = np.mean(rgb[y - quadrant_size: y + 1, x - quadrant_size: x + 1, 1])
                rgb[y][x][2] = np.mean(rgb[y - quadrant_size: y + 1, x - quadrant_size: x + 1, 2])
            if min == 2:
                rgb[y][x][0] = np.mean(rgb[y: y + quadrant_size + 1, x - quadrant_size: x + 1, 0])
                rgb[y][x][1] = np.mean(rgb[y: y + quadrant_size + 1, x - quadrant_size: x + 1, 1])
                rgb[y][x][2] = np.mean(rgb[y: y + quadrant_size + 1, x - quadrant_size: x + 1, 2])
            if min == 3:
                rgb[y][x][0] = np.mean(rgb[y: y + quadrant_size + 1, x: x + quadrant_size + 1, 0])
                rgb[y][x][1] = np.mean(rgb[y: y + quadrant_size + 1, x: x + quadrant_size + 1, 1])
                rgb[y][x][2] = np.mean(rgb[y: y + quadrant_size + 1, x: x + quadrant_size + 1, 2])

    return Image.fromarray(rgb.astype(np.uint8))


# Main script
if __name__ == '__main__':

    # Parameter valid for all three filters
    window_size = 9

    # Opening image
    image = Image.open("image5.jpg")
    # Resizing image
    image = image.resize((500, 300))
    # Converting image to numpy array for Mean and Gaussian Filter
    image_mean_gauss = np.array(image, dtype=float)  # image for mean and gaussian filter

    # Mean Filter
    mean_filtered_image = mean_filtering(image_mean_gauss, window_size)
    mean_filtered_image.save("MeanFiltered9_5.jpg")

    # Gaussian Filter (takes extra parameter namely sigma)
    gaussian_filtered_image = gaussian_filtering(image_mean_gauss, window_size, 2)
    gaussian_filtered_image.save("GaussianFiltered9_2_5.jpg")

    # Kuwahara Filter
    kuwahara_filtered_image = kuwahara_filtering(image, window_size)
    kuwahara_filtered_image.save("KuwaharaFiltered9_5.jpg")
