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
        cv2.imwrite(f"cut_test/{indx}.png", mimg)
        #plt.imshow(mimg)
        #plt.show()
        print(f"cut_test/{indx}.png")
        indx += 1
        val_idxs = new_find_value(mimg, last)
        keyarr = []
        valarr = []
        print(val_idxs.keys())
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
        print(direction)

        last = set_last(direction)
        h, w = new_calculate_new_points(h, w, height, width, cstep*3, val_idxs[direction], direction)
        print(h, w, prev_h, prev_w)
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
    print(smallest)
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


def cut_image_from_working_tile(myimg, last):
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
        print(res_arr)
        arr_idx, smallest = find_minimal(res_arr)
        res_dict[DOWN] = smallest
        slice = myimg[tsize:(2*tsize), 0:tsize, :]
        res_dict[LEFT] = analyze_slice(slice)
        slice = myimg[tsize:(2 * tsize), (2*tsize):(3*tsize), :]
        res_dict[RIGHT] = analyze_slice(slice)
        drctn = find_minimal_dict(res_dict)
        print(drctn)
        print(res_dict)
        if drctn == DOWN:
            slice = myimg[(2*tsize):(3*tsize), (arr_idx*step):(arr_idx*step + tsize), :]
            return myimg[(2*tsize):(3*tsize), (arr_idx*step):(arr_idx*step + tsize), :], 2*tsize, arr_idx*step, drctn
        elif drctn == LEFT:
            return myimg[tsize:(2*tsize), 0:tsize, :], tsize, 0, drctn
        elif drctn == RIGHT:
            return myimg[tsize:(2 * tsize), (2*tsize):(3*tsize), :], tsize, 2*tsize, drctn
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
        print(drctn)
        print(res_dict)
        if drctn == UP:
            return myimg[0:tsize, (arr_idx*step):(arr_idx*step + tsize), :], 0, arr_idx*step, drctn
        elif drctn == LEFT:
            return myimg[tsize:(2*tsize), 0:tsize, :], tsize, 0, drctn
        elif drctn == RIGHT:
            return myimg[tsize:(2 * tsize), (2*tsize):(3*tsize), :], tsize, 2*tsize, drctn
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
        print(drctn)
        print(res_dict)
        if drctn == LEFT:
            return myimg[(arr_idx*step):(arr_idx*step + tsize), 0:tsize, :], arr_idx*step, 0, drctn
        elif drctn == UP:
            return myimg[0:tsize, tsize:(2*tsize), :], 0, tsize, drctn
        elif drctn == DOWN:
            return myimg[(2*tsize):(3*tsize), tsize:(2 * tsize), :], 2*tsize, tsize, drctn
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
        print(drctn)
        print(res_dict)
        if drctn == RIGHT:
            return myimg[(arr_idx*step):(arr_idx*step + tsize), (2*tsize):(3*tsize), :], arr_idx*step, 2*tsize, drctn
        elif drctn == UP:
            return myimg[0:tsize, tsize:(2*tsize), :], 0, tsize, drctn
        elif drctn == DOWN:
            return myimg[(2*tsize):(3*tsize), tsize:(2 * tsize), :], 2*tsize, tsize, drctn


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

    cstep = tile_width

    hmax = int(height/100)
    wmax = int(width/100)

    cutting = True
    last = DOWN
    h = x
    w = y
    indx=1

    coordlist = []
    save_img = img[h:(h+cstep), w:(w+cstep), :]
    mask[h:(h + cstep), w:(w + cstep), :] = save_img

    cv2.imwrite(f"cut_test/{indx}.png", save_img)
    print(f"cut_test/{indx}.png")

    if h < cstep:
        h = cstep
    if w < cstep:
        w = cstep
    if height - 2 * cstep < h:
        h = height - 2 * cstep
    if width - 2 * cstep < w:
        w = width - 2 * cstep

    print(f"h: {h}  w: {w}")

    while cutting:
        prev_h = h
        prev_w = w
        # print(h, h+cstep, w, w+cstep)
        coordlist.append((prev_h, prev_w))

        mimg = img[(h-cstep):(h + 2*cstep), (w-cstep):(w + 2*cstep), :]
        print(mimg.shape)
        #plt.imshow(mimg)
        #plt.show()
        save_img, h0, w0, direction = cut_image_from_working_tile(mimg, last)
        print("cut image")
        #plt.imshow(save_img)
        #plt.show()
        w += w0 - cstep  # UWAGA: nasz układ wsp zaczyna się h - cstep; obraz zaczyna się od 0 - nanoszę poprawkę
        h += h0 - cstep
        if w < cstep:
            w = cstep
        if h < cstep:
            h = cstep
        if height - 2 * cstep < h:
            h = height - 2 * cstep
        if width - 2 * cstep < w:
            w = width - 2 * cstep
        mask[h:(h + cstep), w:(w + cstep), :] = save_img
        cv2.imwrite(f"cut_test/{indx}.png", save_img)
        #plt.imshow(mimg)
        #plt.show()
        print(f"cut_test/{indx}.png")
        indx += 1

        last = direction
        print(h, w, prev_h, prev_w)
        cutting = check_if_repeats(coordlist, h, w, cstep)
        print(f"h: {h}  w: {w}, height: {height}, width: {width}")
        #plt.imshow(mimg)
        #plt.show()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()
    plt.imshow(mimg)
    plt.show()


if __name__ == "__main__":
    img = cv2.imread(f'krzywy.tiff')
    mask = np.zeros(img.shape, dtype=np.int)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x, y = find_begin(rgb_img, step=100)
    cut2(img, mask, x, y, 200)