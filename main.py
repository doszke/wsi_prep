from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np


def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 70)
    ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
    return ret, mask


def find_values(slice):
    return [x for x in range(np.shape(slice)[1]) if slice[0, x] > 0]


def straight(img, iter=1000):  # todo wybór ilości iteracji
    height, width, _ = np.shape(img)
    _, mask = edge_detection(img)
    mask = np.array(mask)
    plt.imshow(mask)
    #plt.show()
    res = np.zeros(img.shape, dtype=int) + 255
    centers = []
    lens = []
    for x in range(height):
        slice = mask[x:(x+1), :]
        vals = find_values(slice)
        if len(vals) != 0:
            min_v = min(vals)
            max_v = max(vals)
            centers.append(min_v + int((max_v - min_v) / 2))
            lens.append(max_v - centers[x])
        else:
            centers.append(0)
            lens.append(0)
    main_center = max(lens)
    for x in range(len(centers)):
        center = centers[x]
        lenn = lens[x]
        offset = main_center - lenn
        pre_x_start = center - lenn
        pre_x_end = center + lenn
        post_x_start = offset
        post_x_end = offset + (pre_x_end - pre_x_start)
        slice = img[x:(x+1), pre_x_start:pre_x_end]
        res[x:(x+1), post_x_start: post_x_end] = slice
    plt.imshow(res)
    #plt.show()
    return res


if __name__ == '__main__':
    for x in range(1, 6):
        img = cv2.imread('./samples/%d.png' % x)
        resimg = straight(img)
        cv2.imwrite('./results/%d.jpg' % x, resimg)
