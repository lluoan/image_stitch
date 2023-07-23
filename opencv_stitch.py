#%%
import numpy as np
import cv2
from skimage import io
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
from pathlib import Path
import json
import matplotlib.patches as patches
import time
%matplotlib widget

# 读取JSON文件并解析变量
image = io.imread("img/0.jpg")
with open("img/tactile_vertice.json", "r") as f:
    data = json.load(f)
    ncol = data["ncol"]
    nrow = data["nrow"]
    x_0 = data["x_0"]
    y_0 = data["y_0"]
    x_end = data["x_end"]
    y_end = data["y_end"]
    w = data["w"]
    h = data["h"]

rect = []
for i in range(ncol):
    row = []
    for j in range(nrow):
        center_x = int(x_0 + j * (x_end - x_0) / (nrow - 1) if nrow > 1 else x_0 + (x_end - x_0) / 2)
        center_y = int(y_0 + i * (y_end - y_0) / (ncol - 1) if ncol > 1 else y_0 + (y_end - y_0) / 2)
        x1 = int(center_x - w / 2)
        y1 = int(center_y - h / 2)
        x2 = int(center_x + w / 2)
        y2 = int(center_y + h / 2)
        row.append(((x1, y1), (x2, y2), (center_x, center_y)))
    rect.append(row)
    
fig,ax = plt.subplots(figsize=(10,10))

# Display the image
ax.imshow(image, cmap='gray')
ax.axis('on')

k = 1

for i in range(ncol):
    for j in range(nrow):
        (x1, y1), (x2, y2), _ = rect[i][j]  # 获取当前矩形的左下角、右上角坐标

        left_top = (x1, y1)
        right_bottom = (x2, y2)
        
        # Create a Rectangle patch
        rect_patch = patches.Rectangle((left_top[0],left_top[1]),
                                right_bottom[0]-left_top[0],right_bottom[1] - left_top[1],
                                linewidth=2, edgecolor='w',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect_patch)

        # cut roi tiles and save
        img_tile = image[int(left_top[1]):int(right_bottom[1]), int(left_top[0]):int(right_bottom[0]), :]
        # plt.figure()
        # plt.title(k)
        # plt.imshow(img_tile)
    #     plt.imshow(cv2.flip(img_tile, -1))
        k = k + 1

#%%
vstack = []
tile_list = []

for i in range(ncol):
    hstack=[]
    for j in range(nrow):
        (x1, y1), (x2, y2), _ = rect[i][j]  # 获取当前矩形的左下角、右上角坐标

        left_top = (x1, y1)
        right_bottom = (x2, y2)

        tile = image[int(left_top[1]):int(right_bottom[1]),
                    int(left_top[0]):int(right_bottom[0]), :]
        tile_flipped = cv2.flip(tile, -1)
        
        tile_list.append(tile_flipped)
        if hstack == []:
            hstack = tile
        else:
            hstack = np.hstack((hstack, tile))
    if vstack == []:
        vstack = hstack
    else:
        vstack = np.vstack((vstack, hstack))


plt.figure(figsize=(7,7))
plt.imshow(vstack, cmap='gray')


#%%
# img1 = rgb2gray(tile_list[0])
# img2 = rgb2gray(tile_list[1])
# img1 = rgb2hsv(tile_list[0])
# img2 = rgb2hsv(tile_list[4])
img1 = tile_list[0]
img2 = tile_list[1]

# Read the images from your directory
image1 = tile_list[0]
image2 = tile_list[1]

# Create a list of images
images = []
images.append(image1)
images.append(image2)

# Create a Stitcher class object with mode panoroma
stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

# Perform the stitching
ret, pano = stitcher.stitch(images)

# Check if stitching was successful
if ret == cv2.Stitcher_OK:
    # Save the result
    cv2.imwrite('output.jpg', pano)
    # Show the result
    cv2.imshow('Panorama', pano)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print("Error during stitching, error code = %d" % ret)

# %%
