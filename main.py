import cv2
import numpy as np
import matplotlib.pyplot as plt

import time

from helpers import (find_seam,
                     rm_seam,
                     duplicate_seam,
                     calc_energy_e1,
                     calc_cme,
                     retarget
                     )


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwrite(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# =========================Fig 5===========================
def fig5():
    print('fig5 step')
    fig5 = imread("./images/fig5.png")
    fig5_narrow = fig5.copy()
    for i in range(350):
        if (i+1) % 10 == 0:
            print(str(i+1) + " seams removed")
        seam = find_seam(fig5_narrow)
        fig5_narrow = rm_seam(fig5_narrow, seam)
    imwrite("out/fig5_rm_final.png", fig5_narrow)


# =========================Fig 8===========================
def enlarge(img, by):
    h, w, _ = img.shape

    add_seams = int(w * by)

    img_wider = img.copy()
    img_w_seams = img.copy()
    img_fordel = img.copy()

    original_inds = list(range(w))
    seams_starts = np.array([])

    print "adding {} seams".format(add_seams)

    for i in range(add_seams):
        seam = find_seam(img_fordel)
        img_fordel = rm_seam(img_fordel, seam)

        seam_ind = seam[-1][1]  # bottom pixel's x coord
        orig_ind = original_inds.pop(seam_ind)

        n_seams_before = len(seams_starts[seams_starts < orig_ind])

        seams_starts = np.append(seams_starts, orig_ind)

        offset = orig_ind + n_seams_before - seam_ind

        # print offset, seam_ind, orig_ind, n_seams_before

        adj_seam = (seam + np.array([0, offset])).tolist()
        adj_seam = [tuple(a) for a in adj_seam]

        img_w_seams = duplicate_seam(img_w_seams, adj_seam, interpolate=False)
        img_wider = duplicate_seam(img_wider, adj_seam)

    return img_wider, img_w_seams


def fig8():
    print("fig8 step")
    fig8 = imread("./images/fig8.png")
    fig8_wider, fig8_w_seams = enlarge(fig8, .4)
    fig8_widerer, _ = enlarge(fig8_wider, .4)

    imwrite('out/fig8_wider.png', fig8_wider)
    imwrite('out/fig8_w_seams.png', fig8_w_seams)
    imwrite('out/fig8_widerer.png', fig8_widerer)


# ====================================Fig 7==========================
def color_T(T):
    T_norm = T.copy()
    T_norm = T_norm.astype(np.float32)
#     cv2.normalize(T, T_norm, 0, 255, cv2.NORM_MINMAX)
    T_norm *= (255.0/T_norm.max())
    T_norm = T_norm.astype(np.uint8)
    T_colored = cv2.applyColorMap(T_norm, cv2.COLORMAP_JET)
    T_colored = cv2.cvtColor(T_colored, cv2.COLOR_BGR2RGB)
    return T_colored


def resize_optimally(images, direction, T, py=.5, px=.7):
    h, w, _ = images[(0, 0)].shape
    new_h = int(h * py)
    new_w = int(w * px)

    r = h - new_h
    c = w - new_w

    opt_path = color_T(T)
    i = r
    j = c
    while not (i == 0 and j == 0):
        opt_path[i, j] = [255, 255, 255]
        if direction[i, j]:
            # next is vertical
            j -= 1
        else:
            # next is horizontal
            i -= 1

    return images[(r, c)], opt_path


def fig7():
    print("fig7 step")
    fig7 = imread("./images/fig7.png")
    # uncomment next line to calculate retargeting on smaller image
    # fig7 = imread("./images/fig7_s.png")

    T, direction, images = retarget(fig7)
    img, opt_path = resize_optimally(images, direction, T)

    imwrite('out/fig7_img.png', img)
    imwrite('out/fig7_path.png', opt_path)


# ======================Exec========================

if __name__ == "__main__":
    # comment those steps that you wish to skip
    fig5()
    fig8()
    fig7()
