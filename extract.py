import os
from skimage import io
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


def new_calculate_new_points(h, w, mh, mw, step, points, direction):
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

def find_begin(img, step=20):
    height, width, _ = img.shape
    relative_white = 235
    if height > width:
        bound = width
    else:
        bound = height
    dimg = int(bound/step)
    for n in range(step):
        nimg = n * dimg
        ver_slice = img[nimg:(nimg+1), :, :]
        hor_slice = img[:, nimg:(nimg+1), :]
        ver_avg = np.average(ver_slice, axis=2)
        hor_avg = np.average(hor_slice, axis=2)
        for i in range(len(ver_avg[0])):
            cell = ver_avg[0, i]
            if cell < relative_white:
                return nimg, i
        for i in range(len(hor_avg)):
            cell = hor_avg[i]
            if cell < relative_white:
                return i, nimg


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

def cut(img, mask, x, y, tile_width):
    """
    wycina ładnie wężyka wzdłuż danych, ale lubi stać przy krawędzi, przez co sporo białego jest
    :param img:
    :param mask:
    :param x:
    :param y:
    :param tile_width:
    :return:
    """
    height, width, _ = img.shape

    cstep = tile_width

    hmax = int(height/100)
    wmax = int(width/100)

    cutting = True
    last = LEFT
    h = x
    w = y

    coordlist = []
    prev_h = 0
    prev_w = 0
    indx=1
    while cutting:
        #print(h, h+cstep, w, w+cstep)
        prev_h = h
        prev_w = w
        coordlist.append((prev_h, prev_w))
        mimg = img[h:(h + cstep), w:(w + cstep), :]
        #plt.imshow(mimg)
        #plt.show()
        # print(f"cut_test/{indx}.png")
        indx += 1
        val_idxs = new_find_value(mimg, last)
        keyarr = []
        valarr = []
        # print(val_idxs.keys())
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
        priviledged_dir = []
        if keyarr.__contains__(DOWN):
            testvals = val_idxs[DOWN]
            if testvals[1] - testvals[0] > tile_width * 0.4:
                priviledged_dir.append(DOWN)
        if keyarr.__contains__(RIGHT):
            testvals = val_idxs[RIGHT]
            if testvals[1] - testvals[0] > tile_width * 0.4:
                priviledged_dir.append(RIGHT)
        if len(priviledged_dir) == 0:
            direction = keyarr[idx]
        else:
            direction = priviledged_dir[0]
        # print(direction)

        last = set_last(direction)
        h, w = new_calculate_new_points(h, w, height, width, cstep*3, val_idxs[direction], direction)
        # print(h, w, prev_h, prev_w)
        if abs(prev_h - h) < cstep * 0.1 and abs(prev_w - w) < cstep * 0.1:
            cutting = False

def analyze_slice(slice):
    """
    zwraca ilość białych pixeli na ramce
    :param slice:
    :return:
    """
    tsize = slice.shape[0]
    edges = [
        np.average(slice[0, 1:(tsize - 1), :], axis=1),
        np.average(slice[tsize - 1, 1:(tsize - 1), :], axis=1),
        np.average(slice[1:(tsize - 1), 0, :], axis=1),
        np.average(slice[1:(tsize - 1), tsize - 1, :], axis=1),
    ]
    score = 0
    for edge in edges:
        score += len([i for i in edge if i > 225])
    return score


def find_minimal(marr):
    """
    zwraca indeks najmniejszej wartości
    :param mdict:
    :return:
    """
    smallest = 10000
    idx = 0
    mmin = min(marr)
    for i in range(len(marr)):
        if marr[i] < smallest:
            smallest = marr[i]
            idx = i
    # print(smallest)
    return idx, smallest


def find_minimal_dict(mdict):
    keys = mdict.keys()
    smallest = 1000000
    smkey = DOWN
    for key in keys:
        if mdict[key] < smallest:
            smallest = mdict[key]
            smkey = key
    return smkey


def cut_image_from_working_tile(myimg, mymask, last):
    """

    :param mimg:
    :param last:
    :return:
    """
    tsize = int(myimg.shape[0]/3)
    nsteps = 10
    step = int(tsize/nsteps)
    res_dict = {}
    if last == DOWN:
        res_arr = []
        for i in range(nsteps):
            slice = myimg[(2*tsize):(3*tsize), (i*step):(i*step + tsize), :]
            res_arr.append(analyze_slice(slice))
        # print(res_arr)
        arr_idx, smallest = find_minimal(res_arr)
        res_dict[DOWN] = smallest
        slice = myimg[tsize:(2*tsize), 0:tsize, :]
        res_dict[LEFT] = analyze_slice(slice)
        slice = myimg[tsize:(2 * tsize), (2*tsize):(3*tsize), :]
        res_dict[RIGHT] = analyze_slice(slice)
        drctn = find_minimal_dict(res_dict)
        # print(drctn)
        # print(res_dict)
        if drctn == DOWN:
            # slice = myimg[(2*tsize):(3*tsize), (arr_idx*step):(arr_idx*step + tsize), :]
            return myimg[(2*tsize):(3*tsize), (arr_idx*step):(arr_idx*step + tsize), :], mymask[(2*tsize):(3*tsize), (arr_idx*step):(arr_idx*step + tsize), :], 2*tsize, arr_idx*step, drctn
        elif drctn == LEFT:
            return myimg[tsize:(2*tsize), 0:tsize, :], mymask[tsize:(2*tsize), 0:tsize, :], tsize, 0, drctn
        elif drctn == RIGHT:
            return myimg[tsize:(2 * tsize), (2*tsize):(3*tsize), :], mymask[tsize:(2 * tsize), (2*tsize):(3*tsize), :], tsize, 2*tsize, drctn
    elif last == UP:
        res_arr = []
        for i in range(nsteps-1):
            slice = myimg[0:tsize, (i*step):(i*step + tsize), :]
            res_arr.append(analyze_slice(slice))
        arr_idx, smallest = find_minimal(res_arr)
        res_dict[UP] = smallest
        slice = myimg[tsize:(2*tsize), 0:tsize, :]
        res_dict[LEFT] = analyze_slice(slice)
        slice = myimg[tsize:(2 * tsize), (2*tsize):(3*tsize), :]
        res_dict[RIGHT] = analyze_slice(slice)
        drctn = find_minimal_dict(res_dict)
        # print(drctn)
        # print(res_dict)
        if drctn == UP:
            return myimg[0:tsize, (arr_idx*step):(arr_idx*step + tsize), :], mymask[0:tsize, (arr_idx*step):(arr_idx*step + tsize), :], 0, arr_idx*step, drctn
        elif drctn == LEFT:
            return myimg[tsize:(2*tsize), 0:tsize, :], mymask[tsize:(2*tsize), 0:tsize, :], tsize, 0, drctn
        elif drctn == RIGHT:
            return myimg[tsize:(2 * tsize), (2*tsize):(3*tsize), :], mymask[tsize:(2 * tsize), (2*tsize):(3*tsize), :], tsize, 2*tsize, drctn
    elif last == LEFT:
        res_arr = []
        for i in range(nsteps-1):
            slice = myimg[(i*step):(i*step + tsize), 0:tsize, :]
            res_arr.append(analyze_slice(slice))
        arr_idx, smallest = find_minimal(res_arr)
        res_dict[LEFT] = smallest
        slice = myimg[0:tsize, tsize:(2*tsize), :]
        res_dict[UP] = analyze_slice(slice)
        slice = myimg[(2*tsize):(3*tsize), tsize:(2 * tsize), :]
        res_dict[DOWN] = analyze_slice(slice)
        drctn = find_minimal_dict(res_dict)
        # print(drctn)
        # print(res_dict)
        if drctn == LEFT:
            return myimg[(arr_idx*step):(arr_idx*step + tsize), 0:tsize, :], mymask[(arr_idx*step):(arr_idx*step + tsize), 0:tsize, :], arr_idx*step, 0, drctn
        elif drctn == UP:
            return myimg[0:tsize, tsize:(2*tsize), :], mymask[0:tsize, tsize:(2*tsize), :], 0, tsize, drctn
        elif drctn == DOWN:
            return myimg[(2*tsize):(3*tsize), tsize:(2 * tsize), :], mymask[(2*tsize):(3*tsize), tsize:(2 * tsize), :], 2*tsize, tsize, drctn
    elif last == RIGHT:
        res_arr = []
        for i in range(nsteps-1):
            slice = myimg[(i*step):(i*step + tsize), (2*tsize):(3*tsize), :]
            res_arr.append(analyze_slice(slice))
        arr_idx, smallest = find_minimal(res_arr)
        res_dict[RIGHT] = smallest
        slice = myimg[0:tsize, tsize:(2*tsize), :]
        res_dict[UP] = analyze_slice(slice)
        slice = myimg[(2*tsize):(3*tsize), tsize:(2 * tsize), :]
        res_dict[DOWN] = analyze_slice(slice)
        drctn = find_minimal_dict(res_dict)
        # print(drctn)
        # print(res_dict)
        if drctn == RIGHT:
            return myimg[(arr_idx*step):(arr_idx*step + tsize), (2*tsize):(3*tsize), :], mymask[(arr_idx*step):(arr_idx*step + tsize), (2*tsize):(3*tsize), :], arr_idx*step, 2*tsize, drctn
        elif drctn == UP:
            return myimg[0:tsize, tsize:(2*tsize), :], mymask[0:tsize, tsize:(2*tsize), :], 0, tsize, drctn
        elif drctn == DOWN:
            return myimg[(2*tsize):(3*tsize), tsize:(2 * tsize), :], mymask[(2*tsize):(3*tsize), tsize:(2 * tsize), :], 2*tsize, tsize, drctn


def check_if_repeats(coordlist, curr_h, curr_w, cstep):
    nh = int(curr_h/cstep)
    nw = int(curr_w/cstep)
    for c in coordlist:
        coord_h, coord_w = c
        coord_nh = int(coord_h/cstep)
        coord_nw = int(coord_w/cstep)
        if coord_nh == nh and coord_nw == nw:
            return False
    return True


def cut2(img, mask, x, y, tile_width):
    """
    domyślnie wycina 3x większe kafle robocze, potem wycina ze środka kafelke o właściwym wymiarze; przesuwa się o długość kafelki oczekiwanej
    :param img:
    :param mask: TODO póki co to jest czarne tło + poukładane wycinki; zrobić, by zapisywało maski
    :param x:
    :param y:
    :param tile_width:
    :return:
    """
    # wycinamy pierwszą kafelkę o wsp początkowych + cstep
    # jeżeli w lub h < cstep, ustaw zmienną na cstep
    # PĘTLA:
    # analizyjemy kwadrat o wsp [(h - cstep), (w-cstep)] X [(h + 2cstep), (w + 2cstep)]
    # kwadrat dzielimy an 9 sektorów kwadratowych, każdy o długości cstep (h-cstep do h; h do h+cstep; h+cstep do
    #       h+2cstep, to samo z _w)na środku kwadratu (początek h, w) mamy ostatnio wyciętą tkankę
    # w zależności od kierunku ruchu, wykluczamy  3, leżace na 1 linii komórki (np. dla ruchu w dół,
    #       wszystkie kwadraty o wsp h-cstep do h są wykluczane (tamte rejony oznaczają cofanie sie))
    # pozostałe 5 otaczających sprawdzamy pod kątem ile białego na brzegach - wybieramy kafleke z min(white)
    # wycinamy, zapisujemy
    # analiza wg brzegu odnośnie kierunku, z priorytetem na DÓŁ i PRAWO
    # aktualizujemy wsp h i w względem nowego wycinka (albo h albo w bęzie większe/mniejsze o cstep -
    #       zapobieganie nachodzeniu)
    # analiza współczynników - jak niewielka zmiana, to ustaw cutting na false
    height, width, _ = img.shape
    # print(img.shape)
    cstep = tile_width

    hmax = int(height/100)
    wmax = int(width/100)

    cutting = True
    last = DOWN
    h = x
    w = y
    # print(w)
    # print(cstep)
    # print(w < cstep)
    if h < cstep:
        h = cstep
    if w < cstep:
        # print("hi lower w")
        w = cstep
    if height - 2 * cstep - 1 < h:
        h = height - 2 * cstep - 1
    if width - 2 * cstep - 1 < w:
        # print("hi higher w")
        w = width - 2 * cstep - 1

    indx=1

    coordlist = []
    save_img = img[h:(h+cstep), w:(w+cstep), :]
    save_mask = mask[h:(h+cstep), w:(w+cstep), :]

    # print(f"before w: {w}")
    res_img = [save_img]
    res_mask = [save_mask]
    while cutting:
        prev_h = h
        prev_w = w
        # print(h, h+cstep, w, w+cstep)
        coordlist.append((prev_h, prev_w))
        # print(w)
        # print(cstep)
        # print(w-cstep)
        # print(w + 2*cstep)
        mimg = img[(h-cstep):(h + 2*cstep), (w-cstep):(w + 2*cstep), :]
        __w, __h, _ = mimg.shape
        # print(mimg.shape)
        assert __w == __h # mimg
        mmask = mask[(h-cstep):(h + 2*cstep), (w-cstep):(w + 2*cstep), :]
        # print(mimg.shape)
        save_img, save_mask, h0, w0, direction = cut_image_from_working_tile(mimg, mmask, last)
        __w, __h, _ = save_img.shape
        assert __w == __h  # save_img
        res_img.append(save_img)
        res_mask.append(save_mask)
        w += w0 - cstep  # UWAGA: nasz układ wsp zaczyna się h - cstep; obraz zaczyna się od 0 - nanoszę poprawkę
        h += h0 - cstep
        if w < cstep:
            w = cstep
        if h < cstep:
            h = cstep
        if height - 2 * cstep - 1 < h:
            # print("hi")
            h = height - 2 * cstep - 1
        if width - 2 * cstep - 1 < w:
            # print("hi")
            w = width - 2 * cstep - 1
        indx += 1

        last = direction
        # print(h, w, prev_h, prev_w, height, width)
        cutting = check_if_repeats(coordlist, h, w, cstep)
        #plt.imshow(mimg)
        #plt.show()
    return res_img, res_mask


def load_data(img_path, mask_path):
    img_train = []
    mask_train = []
    tilesize = 200
    first = True
    for name in os.listdir(img_path):
        part1, part2 = name.split(".")
        img = io.imread(img_path + name)
        hh, ww, _ = img.shape
        if hh < tilesize * 3 or ww < tilesize * 3:
            print(f"pominięto {name} ")
            continue
        # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = io.imread(mask_path + part1 + "_mask." + part2)
        x, y = find_begin(img, step=100)
        tile_img, tile_mask = cut2(img, mask, x, y, tilesize)
        tile_res = [np.max(msk) for msk in tile_mask]
        tile_img_np = np.array(tile_img)
        tile_res_np = np.array(tile_res)

        if first:
            first = False
            img_train = tile_img_np
            mask_train = tile_res_np
        else:
            img_train = np.concatenate((img_train, tile_img_np))
            mask_train = np.concatenate((mask_train, tile_res_np))
    return img_train, mask_train


if __name__ == "__main__":
    import os
    import time

    import h5py

    BASE = "G:/panda/d/dset/"
    IMG = "train_img/"
    MASK = "train_mask/"
    t1 = time.time()
    _counter = 1
    vec = os.listdir(BASE + IMG)
    tilesize = 200
    sizearr = []
    img_train = []
    mask_train = []
    max_mask = []
    with h5py.File(BASE + 'mydataset.hdf5', 'w') as hf:
        for name in os.listdir(BASE + IMG):
            # print(_counter)
            _counter += 1
            part1, part2 = name.split(".")
            #img = cv2.imread(BASE + IMG + name)
            #hh, ww, _ = img.shape
            #if hh < tilesize*3 or ww < tilesize*3:
            ##    print(f"file {name} omitted")
             #   continue
            #rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #mask = cv2.imread(BASE + MASK + part1 + "_mask." + part2)
            #x, y = find_begin(rgb_img, step=100)
            #tile_img, tile_mask = cut2(img, mask, x, y, tilesize)
            #tile_img_np = np.array(tile_img)
            #tile_mask_np = np.array(tile_mask)
            grp = hf.create_group(name)
            #print(tile_img_np.shape)
            #grp.create_dataset("img", data=tile_img_np, shape=tile_img_np.shape, dtype='i', chunks=True)
            #grp.create_dataset("mask", data=tile_mask_np, shape=tile_mask_np.shape, dtype='i', chunks=True)
            #max_mask.append(np.max(tile_mask_np))
            #sizearr.append(len(tile_img))
    t2 = time.time()
    # print(max(max_mask))
    # for x in range(0, 5):
      #  print(len([i for i in max_mask if i == x]))
    plt.hist(max_mask, bins=5)
    plt.show()
    print(f"Time: {t2-t1}")
    #plt.plot(sizearr)
    #plt.show()
    # print(f"sum: {sum(sizearr)}")
    img_train_np = np.array(img_train)
    mask_train_np = np.array(mask_train)
    # print(img_train_np.shape)
    # print(mask_train_np.shape)

