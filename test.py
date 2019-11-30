import cv2
import numpy as np
import matplotlib.pyplot as plt

from helpers import (find_seam,
                     rm_seam,
                     duplicate_seam,
                     calc_energy_e1
                     )


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def test_rm():
    # fig5 = imread("./images/fig5.png")
    # fig5 = fig5[150:400, 50:150, :]
    fig5 = imread("./test_img/fig5_rm_final.png")
    seam = find_seam(fig5)
    fig5_narrow = rm_seam(fig5, seam)

if __name__ == "__main__":
    test_rm()
