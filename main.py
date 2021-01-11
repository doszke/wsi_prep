from scipy import stats
from skimage import io
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 70)
    ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
    return ret, mask


def find_values(slice):
    res = []
    for y in range(slice.shape[1]):
        res.append([z for z in range(np.shape(slice)[0]) if slice[z, y] > 0])
    return res


def mean(img):
    return np.average(img, axis=2)


def local_centers(vals, start, stop):
    values = []
    lefts = []
    rights = []
    args = [a for a in range(start, start + len(vals))]
    n_args = []
    for i in range(len(args)):
        val = vals[i]
        if len(val) != 0:
            left = val[0]
            right = val[len(val) - 1]
            values.append(int((left + (right - left) / 2)))
            lefts.append(left)
            rights.append(right)
            n_args.append(args[i])
    return n_args, values, lefts, rights


def straight(img, iter=10):  # todo wybór ilości iteracji
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    height, width, _ = np.shape(img)
    mask = treshold(img)
    mask = np.array(mask)
    res = np.zeros(img.shape, dtype=int) + 255
    centers = [0] * width
    lefts = [0] * width
    rights = [0] * width
    dt = width / iter
    args = 0
    for x in range(iter):
        _x = int(x * dt)
        _x_1 = int(_x + dt)
        slice = mask[:, _x:_x_1]
        vals = find_values(slice)  # 2d!!!!
        args, l_centers, lef, rig = local_centers(vals, _x, _x_1)
        #print("_" * 100)
        if not len(l_centers) == 0:
            vals = l_centers
            a, b, _, _, _ = stats.linregress(args, vals)
            #print("args: ", args)
            #print("vals: ", vals)
            #print("lef: ", lef)
            #print("centers: ", end='')
            for i in range(len(args)):
                arg = args[i]
                centers[arg] = int(a * arg + b)
                #print(int(a * arg + b), end=", ")
                lefts[arg] = lef[i]
                rights[arg] = rig[i]
            #print()
    main_center = int(res.shape[0]/2)
    for x in range(len(centers)):
        c = centers[x]
        l = lefts[x]
        r = rights[x]
        if l == r == 0:
            continue
        img[c, x, 0] = 0
        img[c, x, 1] = 0
        img[c, x, 2] = 0
        lp = c - l
        rp = r - c
        img_start = l
        img_end = r
        res_start = main_center - lp
        res_end = main_center + rp
        assert (img_end - img_start == res_end - res_start)
        #print(x, img_start, img_end, res_start, res_end)
        res[res_start:res_end, x:(x+1)] = img[img_start:img_end, x:(x+1)]
    #plt.imshow(res)
    #plt.show()

    return res


def rotate_fragment(fragment, angle):
    (h, w) = fragment.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(fragment, matrix, (w, h))


def rotate(img, iter=10):
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    height, width, _ = np.shape(img)
    mask = treshold(img)
    mask = np.array(mask)
    res = np.zeros(img.shape, dtype=int) + 255
    centers = [0] * width
    lefts = [0] * width
    rights = [0] * width
    dt = width / iter
    alpha=0
    for x in range(iter):
        _x = int(x * dt)
        _x_1 = int(_x + dt)
        slice = mask[:, _x:_x_1]
        vals = find_values(slice)  # 2d!!!!
        args, l_centers, lef, rig = local_centers(vals, _x, _x_1)
        # print("_" * 100)
        if not len(l_centers) == 0:
            vals = l_centers
            a, b, _, _, _ = stats.linregress(args, vals)
            alpha = math.atan(a) * 180 / math.pi
            print(f"alpha: {alpha}")
            print(len(lef))
            tlefts = []
            trights = []
            for xx in args:
                idx = xx - args[0]
                print(idx)
                tlefts.append(lef[idx])
                trights.append(rig[idx])
            _y = np.min(tlefts)
            _y_1 = np.max(trights)
            fragment = img[_y:_y_1, _x:_x_1, :]
            fig = plt.figure()
            plt.subplot(211)
            plt.imshow(fragment)
            rotated = rotate_fragment(fragment, alpha)
            plt.subplot(212)
            plt.imshow(rotated)
            plt.show()



def treshold(img):
    mask = mean(img)
    print(mask.shape)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] >= 230:
                mask[x, y] = 0  # remove all 255's
            else:
                mask[x, y] = 255
    mask = cv2.erode(mask, np.ones([3,3]))
    mask = cv2.dilate(mask, np.ones([3,3]))
    return mask


if __name__ == '__main__':
    #img = io.MultiImage('samples/1.png')
    img = cv2.imread('samples/1.png')
    print(img.data)
    rotate(img)

