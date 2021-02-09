import numpy as np


class Cutter:

    def get_patch_location(self, img):
        height, width, _ = img.shape

        ys = 0
        idx = 0
        while True:
            if idx >= height:
                ys = 0
                break
            line = img[idx, ::, :]
            if self.contains_only_white(line):
                idx += 10
            else:
                ys = idx - 10
                if ys < 0:
                    ys = 0
                break

        ye = 0
        idx = height - 1
        while True:
            if idx < 0:
                ye = height - 1
                break
            line = img[idx, ::, :]
            if self.contains_only_white(line):
                idx -= 10
            else:
                ye = idx + 10
                if ye > height:
                    ye = height - 1
                break
        xs = 0
        idx = 0
        while True:
            if idx >= width:
                xs = 0
                break
            line = img[::, idx, :]
            if self.contains_only_white(line):
                idx += 10
            else:
                xs = idx - 10
                if xs < 0:
                    xs = 0
                break
        xe = 0
        idx = width - 1
        while True:
            if idx < 0:
                xe = width - 1
                break
            line = img[::, idx, :]
            if self.contains_only_white(line):
                idx -= 10
            else:
                xe = idx + 10
                if xe > width:
                    xe = width - 1
                break
        print(f"bounds found: {xs} {xe} {ys} {ye}")
        return [xs, xe, ys, ye]


    @staticmethod
    def contains_only_white(v):
        all_white = True
        for x in range(1, len(v)):
            if np.average(v[x]) != 255 and np.average(v[x]) != 0:
                all_white = False
        return all_white

