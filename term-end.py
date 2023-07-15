import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot as plt
import cv2
# https://qiita.com/ITF_katoyu/items/115528c98d2e558c6fc6
import requests
import io
import os
!git clone https://github.com/mdipcit/fip_final
cd fip_final

# 画像の取り込み
img = cv2.imread('noisy_input.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')

src_points = np.float32([[139,249], [1171,149], [286,1091],[1330,801]])
# 変換後の座標を指定
dst_points = np.float32([[0, 0], [1599, 0], [0, 1199], [1599, 1199]])
# 射影変換行列を取得
transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
# 画像を変換
img_transformed = cv2.warpPerspective(img, transform_matrix, (1600, 1200),flags = cv2.INTER_LINEAR)

# 変換後の画像を表示
plt.imshow(img_transformed, cmap='gray')
plt.axis('off')
plt.show()

# 名刺部分のみのヒストグラムの可視化（二値化処理のため）
plt.hist(img_transformed.ravel(),256,[0,256])
plt.show()

# 二値化処理
ret, img_bw = cv2.threshold(img_transformed, 145, 255, cv2.THRESH_BINARY)
plt.imshow(img_bw,cmap="gray")

# メディアンフィルタの適用
filtered_image = cv2.medianBlur(img_bw, 3)
filtered_image = cv2.medianBlur(filtered_image, 3)
plt.imshow(filtered_image,cmap="gray")
