import cv2
import numpy as np
import matplotlib.pyplot as plt

LEFT = "left"
RIGHT = "right"
UP = "up"
DOWN = "down"


def find_value(arr, last):
    mdict = {
        UP: arr[0:1, :][0],
        DOWN: arr[99:100, :][0],
        LEFT: arr[:, 0:1][0],
        RIGHT: arr[:, 99:100][0],
    }
    mdict.pop(last)
    otp = {}
    for key in mdict.keys():
        data = mdict.get(key)
        otp[key] = [i for i in range(len(data)) if data[i] == 255]
    return otp

def new_find_value(mimg, last):
    timg = np.average(mimg, axis=2)
    size = mimg.shape[0]
    mdict = {
        UP: np.reshape(timg[0:1, :], timg.shape[1]),
        DOWN: np.reshape(timg[(size-1):size, :], timg.shape[1]),
        LEFT: np.reshape(timg[:, 0:1], timg.shape[1]),
        RIGHT: np.reshape(timg[:, (size-1):size], timg.shape[1]),
    }
    mdict.pop(last)
    otp = {}
    for key in mdict.keys():
        data = mdict.get(key)
        left = -1
        right = -1
        iter = 0
        istep = 5
        while left < 0:
            if len(data) <= iter:
                left = len(data) - 1
                break
            if data[iter] < 240:
                left = iter
            iter += istep
        iter = len(data) - 1
        while right < 0:
            if 0 > iter:
                right = 0
                break
            if data[iter] < 240:
                right = iter
            iter -= istep
        otp[key] = [left, right]
    return otp

def compare_with_image(mimg, direction_dict):
    mdict = {
        UP: mimg[0:1, 1:99, :][0],
        DOWN: mimg[99:100, 1:99, :][0],
        LEFT: mimg[1:99, 0:1, :][0],
        RIGHT: mimg[1:99, 99:100, :][0],
    }
    # all directions - if not present in 'direction_dict' - they simply won't be used
    to_pop = []
    for direction in direction_dict.keys():
        line = mdict[direction]
        print(direction)
        print(np.average(line))
        plt.imshow(mimg)
        plt.show()
        if np.average(line) > 245:
            to_pop.append(direction)
    for p in to_pop:
        direction_dict.pop(p)
    return direction_dict



def calculate_new_points(h, w, mh, mw, step, direction_dict):
    hstep = step/2
    direction = list(direction_dict.keys())[0] # only one element
    center = int(np.average(direction_dict[direction]))
    if direction == UP:
        point = [h - step, center - hstep]
    elif direction == DOWN:
        point = [h + step, center - hstep]
    elif direction == LEFT:
        point = [center - hstep, w - step]
    else:
        point = [center - hstep, w + step]
    if point[0] < 0 or point[0] > mh:
        point[0] = h
    if point[1] < 0 or point[1] > mw:
        point[1] = w
    return int(point[0]), int(point[1])


def new_calculate_new_points(h, w, mh, mw, step, points):
    hstep = step/2
    center = int((points[0] + points[1])/2)
    if direction == UP:
        point = [h - step, w + center - hstep]
    elif direction == DOWN:
        point = [h + step, w + center - hstep]
    elif direction == LEFT:
        point = [h + center - hstep, w - step]
    else:
        point = [h + center - hstep, w + step]
    if point[0] < 0 or point[0] > mh - step:
        point[0] = h
    if point[1] < 0 or point[1] > mw - step:
        point[1] = w
    return int(point[0]), int(point[1])

def set_last(dir):
    if dir == LEFT:
        return RIGHT
    elif dir == RIGHT:
        return LEFT
    elif dir == UP:
        return DOWN
    else:
        return UP

### version with voronoi
# while cutting:
#     print(h, w, h + 1, w + 1, h * cstep, w * cstep, (h + cstep), (w + cstep))
#     mimg = img[h:(h + cstep), w:(w + cstep), :]
#     mslice = mask[h:(h + cstep), w:(w + cstep)]
#     plt.imshow(mslice)
#     plt.show()
#     val_idxs = find_value(mslice, last)
#     print(val_idxs)
#     length = 0
#
#     direction_dict = {key: val_idxs[key] for key in val_idxs.keys() if len(val_idxs[key]) > 0}
#     direction_dict = compare_with_image(mimg, direction_dict)
#     print(direction_dict)
#     last = set_last(list(direction_dict.keys())[0])
#     h, w = calculate_new_points(h, w, height, width, cstep, direction_dict)


### version without voronoi

kernel = np.ones((5,5),np.uint8)
mask = cv2.imread('ttestt.png', 0)
img = cv2.imread('ttestt.PNG')
height, width, _ = img.shape

cstep = 100

hmax = int(height/100)
wmax = int(width/100)

cutting = True
last = LEFT
h = 0
w = 0


prev_h = 0
prev_w = 0
indx=1
while cutting:
    #print(h, h+cstep, w, w+cstep)
    prev_h = h
    prev_w = w
    mimg = img[h:(h + cstep), w:(w + cstep), :]
    cv2.imwrite(f"cut_test/{indx}.png", mimg)
    indx += 1
    val_idxs = new_find_value(mimg, last)

    keyarr = []
    valarr = []

    for key in val_idxs.keys():
        bonds = val_idxs[key]
        diff = bonds[1] - bonds[0]
        keyarr.append(key)
        valarr.append(diff)
    temp = 0
    idx = 0
    for i in range(len(valarr)):
        if valarr[i] > temp:
            temp = valarr[i]
            idx = i
    direction = keyarr[idx]

    last = set_last(direction)
    h, w = new_calculate_new_points(h, w, height, width, cstep, val_idxs[direction])
    print(h, w, prev_h, prev_w)
    if abs(prev_h - h) < cstep * 0.1 and abs(prev_w - w) < cstep * 0.1:
        cutting = False

