# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:24:39 2022

@author: marco
"""

import numpy as np 
from PIL import Image
from resizeimage import resizeimage

# =============================================================================
# Funcion que convierte imagen a matriz binaria para luego transformarla a un
# array binario
# =============================================================================

img = Image.open('watermark_byn.jpg').convert('L')

np_img = np.array(img)
np_img = ~np_img  # Invierte a B y N
np_img[np_img > 0] = 1

def threshold(col):
    s = sum(col) 
    return int(s > 255 * 3 // 2)

img = Image.open("watermark_byn.jpg")

ratio = float((img.size[1]) / (img.size[0]))

img = resizeimage.resize_cover(img, [100, int(ratio * 100)])

pixels = img.getdata()
binary = list(map(threshold, pixels))

array2d = [binary[i * img.size[0] : (i+1) * img.size[0]] for i in range(img.size[1])]

np_img_array = np_img.flatten(order='C')