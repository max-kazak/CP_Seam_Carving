import cv2
import numpy as np

import time


def calc_energy_e1(image):
    """
    Calculates e1 energy of the image.
    Transforms image to grayscale as a preprocess.
    :param image: source image
    :return: energy matrix
    """
    img = image.copy()

    h, w, d = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT101)

    img = img.astype(np.uint16)

    #Prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dx = cv2.filter2D(img, -1, kernelx)
    dy = cv2.filter2D(img, -1, kernely)

    #Sobel
    # dx = cv2.Sobel(img, cv2.CV_16U, 1, 0, ksize=3)
    # dy = cv2.Sobel(img, cv2.CV_16U, 0, 1, ksize=3)

    energy = abs(dy) + abs(dx)

    energy = energy[:h, :w]

    return energy


def calc_cme(energy, direction='vertical'):
    """
    Calculates cumulative minimum energy to find optimal seam later.

    :param direction: direction in which to calculate cme
    :param energy: energy matrix
    :return: cumulative minimum energy matrix
    """
    if direction == "horizontal":
        energy = energy.T

    h, w = energy.shape
    cme = energy.copy()
    # cme = cme.astype(np.uint32)

    for i, row in enumerate(cme):
        if i == 0:
            continue

        for j, value in enumerate(row):
            if j == 0:
                # left edge
                top_inds = [j, j+1]
            elif j == w-1:
                # right edge
                top_inds = [j-1, j]
            else:
                # middle
                top_inds = [j-1, j, j+1]
            cme[i, j] += min(cme[i-1][top_inds])

    return cme


def find_seam(image, direction="vertical", return_enrg=False):
    """
    Finds seam along _direction_.

    :param image:
    :param direction:
    :return: list of seam pixel coordinates, map of pixels
    """
    if direction == "horizontal":
        image = np.transpose(image, (1, 0, 2))

    energy = calc_energy_e1(image)
    cme = calc_cme(energy)

    h, w = cme.shape

    seam = []
    enrg = 0

    for i in range(h-1, -1, -1):    # iterate from last row to first
        row = cme[i]

        if i == h-1:
            min_cols = list(range(w))   # inds of potential path continuation
        else:
            if seam_col == 0:
                # left edge
                min_cols = [seam_col, seam_col+1]
            elif seam_col == w-1:
                # right edge
                min_cols = [seam_col-1, seam_col]
            else:
                # middle
                min_cols = [seam_col - 1, seam_col, seam_col+1]

        values = row[min_cols]
        min_inds = np.where(values == np.min(values))[0]
        seam_col = min_cols[np.random.choice(min_inds)]

        # seam_col = min_cols[np.argmin(row[min_cols])]    # next path column is the one with min cme

        enrg += energy[i, seam_col]

        if direction == "vertical":
            seam.append((i, seam_col))
        else:
            seam.append((seam_col, i))

    if not return_enrg:
        return seam[::-1]   # return in top-down (left-right) form
    else:
        return seam[::-1], enrg


def rm_seam(image, seam_path, direction='vertical'):
    """
    Remove pixels from the image along the seam

    :param image:
    :param seam_path:
    :param direction: seam direction, vertical or horizontal
    :return: new image
    """
    new_img = image.copy()

    if direction == "horizontal":
        new_img = np.transpose(new_img, (1, 0, 2))

    h, w, d = new_img.shape

    seam_map = np.zeros((h, w, d))
    for p in seam_path:
        if direction == 'vertical':
            seam_map[p[0], p[1], :] = 1
        else:
            seam_map[p[1], p[0], :] = 1

    new_img = new_img[seam_map == 0]

    new_img = new_img.reshape(h, w - 1, d)

    if direction == 'horizontal':
        new_img = np.transpose(new_img, (1, 0, 2))

    return new_img


def duplicate_seam(image, seam_path, direction='vertical', interpolate=True):
    """
    Insert duplicate seam with average values of pixels on sides of the seam.

    :param interpolate: interpolate inserted seam (T) or leave it blank (F)
    :param image:
    :param seam_path:
    :param direction: seam direction, vertical or horizontal
    :return: new image
    """
    h, w, _ = image.shape

    if direction == 'vertical':
        new_img = np.zeros((h, w+1, 3), dtype=np.uint8)
    else:
        new_img = np.zeros((h+1, w, 3), dtype=np.uint8)

    new_h, new_w, _ = new_img.shape

    seam_map = np.zeros((new_h, new_w, 3))
    for p in seam_path:
        seam_map[p[0], p[1], :] = 1

    # copy pixels from original to all non-seam pixels of the new image
    new_img[seam_map == 0] = image.flatten()

    if interpolate:
        if direction == 'vertical':
            for p in seam_path:
                # average seam pixel values with surrounding pixels
                if p[1] == 0:
                    # left edge
                    new_img[p] = new_img[p[0], p[1]+1]
                elif p[1] == new_w-1:
                    # right edge
                    new_img[p] = new_img[p[0], p[1]-1]
                else:
                    # middle
                    pix_delta = np.array([[0, -1], [0, 1]])
                    surrounding_pixels_coords = pix_delta + p
                    new_img[p] = np.mean(new_img[surrounding_pixels_coords.T.tolist()], axis=0)
        else:
            for p in seam_path:
                # average seam pixel values with surrounding pixels
                if p[1] == 0:
                    # top edge
                    new_img[p] = new_img[p[0]+1, p[1]]
                elif p[1] == new_h-1:
                    # bottom edge
                    new_img[p] = new_img[p[0]-1, p[1]]
                else:
                    #middle
                    pix_delta = np.array([[-1, 0], [1, 0]])
                    surrounding_pixels_coords = pix_delta + p
                    new_img[p] = np.mean(new_img[surrounding_pixels_coords.T.tolist()], axis=0)
    else:
        for p in seam_path:
            new_img[p] = [255, 0, 0]

    return new_img


def retarget(image):
    h, w, _ = image.shape

    images = {(0, 0): image.copy()}  # (r,c): image
    direction = np.zeros((h - 1, w - 1), dtype=bool)  # True-vertical, False-horizontal
    T = np.zeros((h - 1, w - 1), dtype=np.uint32)

    i = 0
    total = (h - 1) * (w - 1)
    ts = time.time()

    for r, c in gen_diag_iter_inds(h - 1, w - 1):
        i += 1

        if (r, c) == (0, 0):
            continue

        if r == 0:
            # top edge

            # vertical removal
            img_v = images[(r, c - 1)]
            seam_v, e_v = find_seam(img_v, direction='vertical', return_enrg=True)
            images[(r, c)] = rm_seam(img_v, seam_v, direction='vertical')
            T[r, c] = T[r, c - 1] + e_v
            direction[r, c] = True

        elif c == 0:
            # left edge

            # horizontal removal
            img_h = images[(r - 1, c)]
            seam_h, e_h = find_seam(img_h, direction='horizontal', return_enrg=True)
            images[(r, c)] = rm_seam(img_h, seam_h, direction='horizontal')
            T[r, c] = T[r - 1, c] + e_h
            direction[r, c] = False

        else:
            # middle

            # horizontal removal seam
            img_h = images[(r - 1, c)]
            seam_h, e_h = find_seam(img_h, direction='horizontal', return_enrg=True)

            # vertical removal seam
            img_v = images[(r, c - 1)]
            seam_v, e_v = find_seam(img_v, direction='vertical', return_enrg=True)

            if (T[r - 1, c] + e_h) < (T[r, c - 1] + e_v):
                # removing horizontal seam is more efficient
                images[(r, c)] = rm_seam(img_h, seam_h, direction='horizontal')
                T[r, c] = T[r - 1, c] + e_h
                direction[r, c] = False
            else:
                # removing vertical seam is more efficient
                images[(r, c)] = rm_seam(img_v, seam_v, direction='vertical')
                T[r, c] = T[r, c - 1] + e_v
                direction[r, c] = True

    return T, direction, images


# ==========================not important functions=======================

def gen_diag_iter_inds(h, w):
    yi = np.zeros((h, w))
    for i in range(yi.shape[0]):
        yi[i, :] = i

    xi = np.zeros((h, w))
    for i in range(xi.shape[1]):
        xi[:, i] = i

    diag_coord_y = [yi[::-1, :].diagonal(i) for i in range(-yi.shape[0] + 1, yi.shape[1])]
    diag_coord_x = [xi[::-1, :].diagonal(i) for i in range(-xi.shape[0] + 1, yi.shape[1])]

    diag_coord_y = [item for sublist in diag_coord_y for item in sublist]
    diag_coord_x = [item for sublist in diag_coord_x for item in sublist]

    return [tuple(c) for c in np.array([diag_coord_y, diag_coord_x]).astype(int).T.tolist()]
