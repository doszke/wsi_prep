from scipy import stats
from skimage import io
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import pandas as pd
import os
from tifffile import tifffile as tff
import imutils
from staintools.reinhard_normalization import ReinhardColorNormalizer

sharp = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

reinhard = ReinhardColorNormalizer()

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
            fragment = cv2.resize(fragment, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            rotated = imutils.rotate(fragment, alpha)
            rotated = cv2.resize(rotated, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)

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


sz = 120
N = 20
patients = 5


def tile(img, mask):
    # from https://www.kaggle.com/ayeshasaqib/final-project
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    return img, mask


from img_cutter import Cutter

if __name__ == '__main__':

    TRAIN = 'G:/panda/d/train_images/'
    MASKS = 'G:/panda/d/train_label_masks/'
    TRAIN_OUT = 'G:/panda/d/my_train_images/'
    MASKS_OUT = 'G:/panda/d/my_train_labels/'

    img_ids = os.listdir(TRAIN)
    mask_ids = os.listdir(MASKS)

    assert len(img_ids) == len(mask_ids)

    cutter = Cutter()
    # 2209
    # checkpoint = 7137
    # for x in range(checkpoint, len(img_ids)):
    #     print(f'{x}/{len(img_ids)-1}')
    #     img_id = img_ids[x]
    #     print(img_id)
    #     mask_id = mask_ids[x]
    #     img = io.MultiImage(os.path.join(TRAIN, img_id))
    #     mask = io.MultiImage(os.path.join(MASKS, mask_id))
    #     img = img[1]
    #     mask = mask[1][:, :, 0]
    #     xs, xe, ys, ye = cutter.get_patch_location(img)
    #     nimg = img[ys:ye, xs:xe]
    #     nmask = mask[ys:ye, xs:xe]
    #     cv2.imwrite(os.path.join(TRAIN_OUT, img_id), cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR))
    #     cv2.imwrite(os.path.join(MASKS_OUT, mask_id), cv2.cvtColor(nmask, cv2.COLOR_RGB2BGR))

    import cv2
    import numpy as np

    #img = cv2.imread(f"samples/1.png")
    #rotate(img)

    for x in range(1, 7):
        img = cv2.imread(f'voronoi_test/{x}.png', 0)
        _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_not(img)
        kernel = np.ones((5, 5), np.uint8)

        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)

        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)

        ret, img = cv2.threshold(img, 127, 255, 0)
        plt.imshow(img)
        plt.show()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while (not done):
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        cv2.imshow("skel", skel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(f"voronoi_output/{x}.png", skel)


