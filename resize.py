# coding: utf-8
import os
from glob import glob
import cv2
import numpy as np

HEIGHT = 10  # 縮小後の縦のサイズ
WIDTH  = 15  # 縮小後の横のサイズ

# ディレクトリ内の画像ファイルの取得
files = glob("static/images/*.png")

for filepath in files:
    # OpenCVで画像の読み込み
    img = cv2.imread(filepath)
    # 画像の縮小
    small_image = cv2.resize(img, (HEIGHT, WIDTH))
    # 縮小後のファイル名
    small_filename = "static/small_images/" + os.path.basename(filepath)
    # 画像の保存
    cv2.imwrite(small_filename, small_image)
