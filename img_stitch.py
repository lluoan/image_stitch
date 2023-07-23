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
        # tile_list.append(tile)

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
# img1 = img1.astype(np.float32)
# img2 = img2.astype(np.float32)
# scale image brightness
# img2[:,:, 2] = img2[:,:, 2]*1.1

# img1 = hsv2rgb(img1)*255
# img2 = hsv2rgb(img2)*255

fig, ax = plt.subplots(1,2, figsize=(8, 5))
ax = ax.ravel()

ax[0].imshow(img1[:,:, :].astype(np.uint8), cmap='gray')
ax[1].imshow(img2[:,:, :].astype(np.uint8), cmap='gray')
print(img1.shape, img2.shape)


#%%
# ol_w = [6,8,7,9,5,7,6] ground truth
# assume that the overlapping is around 10% of roi size
dim = 1
nominal_ov_pos = 100
bound = nominal_ov_pos-2 # +- bounds around nominal_ov_pos

ov_position = [i for i in range(nominal_ov_pos - bound, nominal_ov_pos+bound+1)]

def find_ov(img1, img2, nominal_ov, bound_ov, dim=1):
    '''
        default dimension is along axis 1 (width of the images)
    '''
    cross_corr = []
    tested_ov = range(nominal_ov - bound_ov, nominal_ov+bound_ov + 1)

    for ov in tested_ov:
        if dim == 1:
            ov_area1 = img1[:, -ov:,:]
            ov_area2 = img2[:,  :ov,:]
        elif dim == 0:
            ov_area1 = img1[-ov:, :,:]
            ov_area2 = img2[:ov,  :,:]
#         print(ov_area1.shape)
#         print(ov_area2.shape)
#         print(ov_area1[:,:,2].shape)
#         a = ov_area1[:,:,i]/1.0
#         b = ov_area2[:,:, i]/1.0
# #         print(a.shape, b.shape)
# #         print(a,b)
#         corr = np.sum(np.multiply(a, b))/(np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
        # multi-channel cross correlation
        corr = 0.0
#         print(ov_area1.shape)
        for i in range(3):
            a = ov_area1[:,:,i]/1.0
            b = ov_area2[:,:, i]/1.0
            corr = corr + np.sum(np.multiply(a, b)) / np.sqrt(np.sum(a**2) * np.sum(b**2))
        corr = corr/3.0
#         print(corr)
        # rgb color distance -> minimize
#         print(np.sum(np.square(img1[:, -(ov+1), :] - img2[:, 0, :])) + np.sum(np.square(img1[:,-1,:] - img2[:, ov]) ))
        cross_corr.append(corr)
#         print(corr)
#         print('---')
        
#     print(cross_corr)
    ov_pos = tested_ov[np.argmax(cross_corr)]
    print(max(cross_corr))
    if dim == 1:
        ov_1 = img1[:, -ov_pos:, :]
        ov_2 = img2[:,  :ov_pos, :]
    elif dim == 0:
        ov_1 = img1[-ov_pos:, :, :]
        ov_2 = img2[:ov_pos,  :, :]
    
    return ov_pos, ov_1, ov_2, cross_corr

def find_ov_hsv(img1, img2, nominal_ov, bound_ov, dim=1):
    '''
        default dimension is along axis 1 (width of the images)
    '''
    img1_hsv = cv2.cvtColor(img1.copy(), cv2.COLOR_RGB2HSV)
    img2_hsv = cv2.cvtColor(img2.copy(), cv2.COLOR_RGB2HSV)
    
    cross_corr = []
    tested_ov = range(nominal_ov - bound_ov, nominal_ov+bound_ov + 1)
    for ov in tested_ov:
        if dim == 1:
            ov_area1 = img1_hsv[:, -ov:,:]
            ov_area2 = img2_hsv[:,  :ov,:]
        elif dim == 0:
            ov_area1 = img1_hsv[-ov:, :,:]
            ov_area2 = img2_hsv[:ov,  :,:]
#         print(ov_area1.shape)
#         print(ov_area2.shape)
#         print(ov_area1[:,:,2].shape)
#         a = ov_area1[:,:,i]/1.0
#         b = ov_area2[:,:, i]/1.0
# #         print(a.shape, b.shape)
# #         print(a,b)
#         corr = np.sum(np.multiply(a, b))/(np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
        # multi-channel cross correlation
        corr = 0.0
#         print(ov_area1.shape)
        for i in range(2):
            a = ov_area1[:,:, i]
            b = ov_area2[:,:, i]
            corr = corr + np.sum(np.multiply(a, b)) / np.sqrt(np.sum(a**2) * np.sum(b**2))
        corr = corr/2.0
#         print(corr)
        # rgb color distance -> minimize
#         print(np.sum(np.square(img1[:, -(ov+1), :] - img2[:, 0, :])) + np.sum(np.square(img1[:,-1,:] - img2[:, ov]) ))
        cross_corr.append(corr)
#         print(corr)
#         print('---')
        
#     print(cross_corr)
    ov_pos = tested_ov[np.argmax(cross_corr)]
    print(np.argmax(cross_corr))
    if dim == 1:
        ov_1 = img1[:, -ov_pos:, :]
        ov_2 = img2[:,  :ov_pos, :]
    elif dim == 0:
        ov_1 = img1[-ov_pos:, :, :]
        ov_2 = img2[:ov_pos,  :, :]
    
    return ov_pos, ov_1, ov_2, cross_corr


img1 = img1.astype(np.float32)
img2 = img2.astype(np.float32)
t1 = time.time()
ov_pos, ov1, ov2, cross_corr = find_ov_hsv(img1, img2, nominal_ov_pos, bound, dim=dim)

print(ov_pos)
print(time.time() - t1)


# fig, ax = plt.subplots(1,2, figsize=(8, 5))
# ax = ax.ravel()

# ax[0].imshow(ov1.astype(np.uint8), cmap='gray')
# ax[1].imshow(ov2.astype(np.uint8), cmap='gray')

from matplotlib import collections as matcoll
print(ov_position[int(np.where(cross_corr==np.max(cross_corr))[0])], max(cross_corr))

# ov_position = [3, 4, 5, 6, 7, 8, 9, 10, 11]
# corr = [0.9909849265735179, 0.996767876208091, 0.999480813538509, 0.9965571362709759, 0.9890487794768316, 
#         0.9807278651316307, 0.9727282861279635, 0.9648033317627641, 0.9573152509152888]

# ov_position = range(37 - 10, 37+10+1)

lines = []
for i in range(len(ov_position)):
    pair=[(ov_position[i],0.0), (ov_position[i], cross_corr[i])]
    lines.append(pair)
    
linecoll = matcoll.LineCollection(lines, color='k', linewidths=3)
fig, ax = plt.subplots(figsize=(10,6))
ax.add_collection(linecoll)

# plt.figure()
plt.scatter(ov_position, cross_corr, c='k', s=60)
plt.ylim(min(cross_corr)-0.05, 1)
# plt.xlabel('Overlapping position')
# plt.ylabel('Cross correlation')

plt.tick_params(bottom=False, labelleft=False, left=False, labelsize=14)

#%%
def stitch(img1, img2, ov_pos, ov1, ov2, blend=True, dim=1):
#     ov_avg = (ov1/2+ov2/2).astype('uint8')
    
    t1 = time.time()
    if blend:
        # linear blending
        ov1_blended = ov1.copy()
        ov2_blended = ov2.copy()
        for i in range(ov_pos):
            blend_coef = -(1/(ov_pos-1))*i + 1
            if dim==1:
                ov1_blended[:,i,:] = ov1[:,i,:]*blend_coef
                ov2_blended[:,i,:] = ov2[:,i,:]*(1.-blend_coef)
            elif dim==0:
                ov1_blended[i, :,:] = ov1[i, :,:]*blend_coef
                ov2_blended[i, :,:] = ov2[i, :,:]*(1.-blend_coef)
        ov_blended = ov1_blended + ov2_blended
        ov = ov_blended
    else:
        ov = ov1
#     print('time: linear blending >>>', time.time() - t1)
    
    if dim == 1:
        stitched_img = np.concatenate((img1[:, :-ov_pos, :], ov, img2[:, ov_pos: :]), axis = 1)
    elif dim == 0:
        stitched_img = np.concatenate((img1[:-ov_pos, :, :], ov, img2[ov_pos:, :, :]), axis = 0)
    
    return stitched_img

def image_alignment(img1, img2, ov_pos, dim=1):
    # print(img1.shape, img2.shape)
    height, width1, _ = img1.shape
    _, width2, _ = img2.shape
    if dim == 1:
        blank1 = np.zeros([height, width2-ov_pos, 3]).astype('uint8')
        align_1 = np.concatenate((img1, blank1), axis=1)
        blank2 = np.zeros([height, width1-ov_pos, 3]).astype('uint8')
        align_2 = np.concatenate((blank2, img2), axis=1)
#         print(align_2.shape)
        align_img = np.concatenate((align_1, align_2), axis=0)
    return align_img

# ov_pos, ov1, ov2, c_corr = find_ov_hsv(tile_list[0], tile_list[1], nominal_ov=nominal_ov_pos, bound_ov=bound, dim=1)
# print(ov_pos)        

# 
# ov_pos = 55
plt.figure(figsize=(10, 8))
plt.axis('off')
# print(img1)
print(ov_pos)
stitched_img = stitch(img1, img2, ov_pos, ov1, ov2, blend=1, dim=1)
stitched_img = image_alignment(img1, img2, ov_pos, dim=1)
plt.imshow(stitched_img.astype(np.uint8), cmap='gray')
print(stitched_img.shape)


#%%
def save_image(image,addr,num):
    address = addr + str(num)+ '.jpg'
    cv2.imwrite(address,image)
    
# print(range(2))
t1 = time.time()
img_stitched_height = None
change_sv = 0
s_mean = 28.0
v_mean = 150.0

stitch_calibration = []


for i in range(ncol):
    img_stitched_width = None
    # print(i)
    for j in range(nrow):
#         if i == 0 && j == 0:
        if j == 0:
            if change_sv == False: 
                img_j0 = tile_list[i*ncol]
            else:
                img_j0 = cv2.cvtColor(tile_list[i*ncol], cv2.COLOR_RGB2HSV)
                img_j0 = img_j0.astype(np.float64)
#                 print("b",img_j0.dtype)
                img_j0[:,:,1] = np.clip(img_j0[:,:,1] * s_mean / np.mean(img_j0[:,:,1]),0.0,255.0)
                img_j0[:,:,2] = np.clip(img_j0[:,:,2] * v_mean / np.mean(img_j0[:,:,2]),0.0,255.0)
                img_j0 = cv2.cvtColor(img_j0.astype(np.uint8), cv2.COLOR_HSV2RGB)
#                 print(img_j0.dtype)

            img_stitched_width = img_j0
#             img_stitched_width = tile_list_histMatched[i*4+j]
        else:

            if change_sv == False: 
                img_1 =  tile_list[i*ncol + j-1]#tile_list_histMatched[i*4 + j-1]
                img_2 =  tile_list[i*ncol + j] #tile_list_histMatched[i*4 + j]
            
            else:
                img_1 = cv2.cvtColor(tile_list[i*ncol + j-1], cv2.COLOR_RGB2HSV)
                img_1 = img_1.astype(np.float64)
                img_1[:,:,1] = np.clip(img_1[:,:,1] * s_mean / np.mean(img_1[:,:,1]),0,255)
                img_1[:,:,2] = np.clip(img_1[:,:,2] * v_mean / np.mean(img_1[:,:,2]),0,255)
                img_1 = cv2.cvtColor(img_1.astype(np.uint8), cv2.COLOR_HSV2RGB)

                img_2 = cv2.cvtColor(tile_list[i*ncol + j], cv2.COLOR_RGB2HSV)
                img_2 = img_2.astype(np.float64)
                img_2[:,:,1] = np.clip(img_2[:,:,1] * s_mean / np.mean(img_2[:,:,1]),0,255)
                img_2[:,:,2] = np.clip(img_2[:,:,2] * v_mean / np.mean(img_2[:,:,2]),0,255)
                img_2 = cv2.cvtColor(img_2.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            
#             print(img1.shape, img2.shape)
#             time.sleep(0.1)
#             plt.figure()
#             plt.subplot(1,2,1)
#             plt.imshow(img1)
#             plt.axis('off')
#             plt.subplot(1,2,2)
#             plt.imshow(img2)
#             plt.axis('off')

            ov_pos, ov1, ov2, c_corr = find_ov_hsv(img_1.astype(np.float32), img_2.astype(np.float32), nominal_ov=nominal_ov_pos, bound_ov=bound, dim=1)
            print('row:'+str(np.max(c_corr)))
            # print('{}th overlap size in row {}: '.format(j-1, i), ov_pos)
            
            stitch_calibration.append(["hor {}".format(j-1),ov_pos])

            
            # stitch images with stacking
            img_stitched_width = stitch(img_stitched_width, img_2, ov_pos, ov1, ov2, blend=True, dim=1)

    # # Display the image using Matplotlib
    # plt.imshow(img_stitched_width.astype(np.uint8), cmap='gray')
    # # Add a title to the image
    # plt.title(i)
    # # Show the image
    # plt.show()


#             print(ov_pos)
        
    # plt.figure(figsize=(10, 3))
    # plt.subplot(1,2,1)
    # plt.imshow(img_stitched_width, cmap='gray')
    # plt.axis('off')
    # plt.subplot(1,2,2)
    # plt.imshow(img2)
    # plt.axis('off')
    # print('-----')
    # print('Stitched row  width >>>  ', img_stitched_width.shape[1])
    if i == 0:
        img_stitched_height = img_stitched_width
        stitch_calibration.append(["ver {}".format(i),0])
        pass
    else:
        width = img_stitched_height.shape[1]
        width_new = img_stitched_width.shape[1]
        wdiff = width - width_new
        # print(i, wdiff)
        if wdiff > 0:
            img_stitched_width = np.hstack((img_stitched_width, img_stitched_width[:, -wdiff:]))
        elif wdiff < 0:
            img_stitched_width = img_stitched_width[:, :wdiff]
#         print(img_stitched_height.shape[1], img_stitched_width.shape[1])
        
        ov_pos_h, ov_h1, ov_h2, c_corr = find_ov_hsv(img_stitched_height, img_stitched_width, 
                                         nominal_ov=nominal_ov_pos, bound_ov=bound, dim=0)
        print('col:'+str(np.max(c_corr)))
        stitch_calibration.append(["ver {}".format(i),ov_pos_h])
        
        # print('Row overlapping:', ov_pos_h)
        img_stitched_height = stitch(img_stitched_height, img_stitched_width, 
                                     ov_pos_h, ov_h1, ov_h2, blend=True, dim=0)
        
t = time.time() - t1
print('Time elapsed: ', t)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img_stitched_height,cv2.COLOR_RGB2GRAY), cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_stitched_height.astype(np.uint8), cmap='gray')
# plt.axis('off')
print(stitch_calibration)

np.save("stitch_calibration.npy", np.asarray(stitch_calibration)) 

print(img_stitched_height.shape)
# %%
