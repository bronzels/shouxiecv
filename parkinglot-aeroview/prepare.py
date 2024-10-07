import os
# glob模块是用来查找符合特定规则命名的文件名的“路径+文件名”，其功能就是检索路径
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pickle

import operator
import collections

def cv_imshow(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def plt_imshow(img, cmap=None):
    cmap = 'gray' if len(img.shape) == 2 else cmap
    plt.imshow(img, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# 读入数据
test_images = [cv2.imread(path) for path in glob.glob('test_images/*.jpg')]  # BGR

cv_imshow('test', test_images[1])

test_image = test_images[1]

def select_rgb_white_yellow(image):
    # 过滤背景
    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    # 三个通道内，低于lower和高于upper的部分分别变成0， 在lower-upper之间的值变成255， 相当于mask，过滤背景
    # 保留了像素值在120-255之间的像素值
    white_mask = cv2.inRange(image, lower, upper)
    masked_img = cv2.bitwise_and(image, image, mask=white_mask)
    return masked_img

masked_img = select_rgb_white_yellow(test_image)
cv_imshow('masked_img', masked_img)

# 转成灰度图
gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
cv_imshow('gray_img', gray_img)

low_threshold, high_threshold = 50, 200
edges_img = cv2.Canny(gray_img, low_threshold, high_threshold)
cv_imshow('edges_img', edges_img)

def select_region(image):
    """这里手动选择区域"""
    rows, cols = image.shape[:2]

    # 下面定义6个标定点，这个点的顺序必须让他换成一个区域，如果调整，可能会交叉起来，所以不要动
    pt_1 = [cols*0.06, rows*0.90]    # 左下
    pt_2 = [cols*0.06, rows*0.70]    # 左上
    pt_3 = [cols*0.32, rows*0.51]    # 中左
    pt_4 = [cols*0.6, rows*0.1]      # 中右
    pt_5 = [cols*0.90, rows*0.1]     # 右上
    pt_6 = [cols*0.90, rows*0.90]    # 右下

    vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
    point_img = image.copy()
    if len(point_img.shape) == 2:
        point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2BGR)
    for point in vertices[0]:
        cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
    cv_imshow('point_img', point_img)

    # 定义mask矩阵，只保留点内部的区域
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, [255,255,255])
    cv_imshow('mask', mask)
    roi_image = cv2.bitwise_and(image, mask)
    return roi_image

roi_image = select_region(edges_img)
cv_imshow('roi_image', roi_image)

#roi_image = select_region(test_image)
#cv_imshow('roi_image_color', roi_image)

#霍夫变换找到直线

def hough_lines(image):
    # 输入的图像需要是边缘检测后的结果
    # minLineLength(线的最短长度，比这个短的都被忽略)和MaxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）
    # rho距离精度,theta角度精度,threshold超过设定阈值才被检测出线段
    return cv2.HoughLinesP(image, rho=0.1, theta=np.pi/10, threshold=15, minLineLength=9, maxLineGap=4)

list_of_lines = hough_lines(roi_image)

print(list_of_lines.shape)

# 过滤检测到的直线
def draw_lines(image, lines, make_copy=True):
    # 过滤霍夫变换检测到直线
    if make_copy:
        image = np.copy(image)
    cleaned = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            # 这里是过滤直线，必须保证不能是斜的线，且水平方向不能太长或者太短
            if abs(y2-y1) <=1 and abs(x2-x1) >= 25 and abs(x2-x1) <= 55:
                cleaned.append((x1,y1,x2,y2))
                cv2.line(image, (x1, y1), (x2, y2), [255, 0, 0], 2)
    print("lines detected: ", len(cleaned))
    return image

roi_image_linedrawed = draw_lines(roi_image, list_of_lines)
cv_imshow('roi_image_linedrawed', roi_image_linedrawed)

def identity_blocks(image, lines, make_copy=True):
    if make_copy:
        new_image = image.copy()
    
    # 过滤部分直线
    stayed_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # 这里是过滤直线，必须保证不能是斜的线，且水平方向不能太长或者太短
            if abs(y2-y1) <=1 and abs(x2-x1) >= 25 and abs(x2-x1) <= 55:
                stayed_lines.append((x1,y1,x2,y2))

    # 对直线按照x1排序，这样能让这些线从上到下排列好，这个排序是从第一列的第一条横线，往下走，然后是第二列的第一条横线往下，...
    list1 = sorted(stayed_lines, key=operator.itemgetter(0,1))

    # 找到多个列，相当于每列是一排车
    clusters = collections.defaultdict(list)
    dIndex = 0
    clus_dist = 10   # 每一列之间的那个距离
    for i in range(len(list1) - 1):
        # 看看两条线之间的距离，如果是一列的，那么x1这个距离应该很近，毕竟是同一列上的
        # 如果这个值大于10了，说明是下一列的了，此时需要移动dIndex，这个表示的是第几列
        distance = abs(list1[i+1][0] - list1[i][0])
        if distance <= clus_dist:
            clusters[dIndex].append(list1[i])
            clusters[dIndex].append(list1[i+1])
        else:
            dIndex += 1
    
    # 得到每列停车位的矩形框
    rects = {}
    i = 0
    for key in clusters:
        all_list = clusters[key]
        cleaned = list(set(all_list))
        # 有5个停车位至少
        if len(cleaned) > 5:
            cleaned = sorted(cleaned, key=lambda tup: tup[1])
            avg_y1 = cleaned[0][1]
            avg_y2 = cleaned[-1][1]
            if abs(avg_y2-avg_y1) < 15:
                continue
            avg_x1 = 0
            avg_x2 = 0
            for tup in cleaned:
                avg_x1 += tup[0]
                avg_x2 += tup[2]
            avg_x1 = avg_x1 / len(cleaned)
            avg_x2 = avg_x2 / len(cleaned)

            rects[i] = [avg_x1, avg_y1, avg_x2, avg_y2]
            i += 1
    print('Num Parking Lanes: ', len(rects))

    # 把列矩形画出来
    buff = 7
    for key in rects:
        tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
        tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
        cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 3)
    return new_image, rects

new_image, rects = identity_blocks(test_image, list_of_lines)
cv_imshow('new_image_lanesquare', new_image)

def rect_finetune(image, rects, copy_img=True):
    if copy_img:
        image_copy = image.copy()
    else:
        image_copy = image

    # 下面需要对上面的狂进行坐标微调，让框更加准确
    # 这个框很重要，影响后面停车位的统计，尽量不能有遗漏
    for k in rects:
        if k == 0:
            rects[k][1] -= 10
        elif k == 1:
            rects[k][1] -= 10
            rects[k][3] -= 10
        elif k == 2 or k == 3 or k == 5:
            rects[k][1] -= 4
            rects[k][3] += 13
        #elif k == 7:
        #    rects[k][3] -= 10
        elif k == 6 or k == 8:
            rects[k][1] -= 18
            rects[k][3] += 12
        elif k == 9:
            rects[k][1] += 10
            rects[k][3] += 10
        elif k == 10:
            rects[k][1] += 45
        elif k == 11:
            rects[k][3] += 45

    buff = 8
    for key in rects:
        tup_topLeft = (int(rects[key][0]-buff), int(rects[key][1]))
        tup_botRight = (int(rects[key][2]+buff), int(rects[key][3]))
        cv2.rectangle(image_copy, tup_topLeft, tup_botRight, (0, 255, 0), 3)

    return image_copy, rects

new_image, rects = rect_finetune(test_image, rects)
cv_imshow('new_image_lanesquare_fituned', new_image)

def draw_parking(image, rects, make_copy=True, save=True):
        if make_copy:
            new_image = image.copy()
        else:
            new_image = image

        gap = 15.5
        spot_dict = {}
        tot_spots = 0

        #微调
        adj_x1 = {0: -8, 1:-15, 2:-15, 3:-15, 4:-15, 5:-15, 6:-15, 7:-15, 8:-10, 9:-10, 10:-10, 11:0}
        adj_x2 = {0: 0, 1: 15, 2:15, 3:15, 4:15, 5:15, 6:15, 7:15, 8:10, 9:10, 10:10, 11:0}
        fine_tune_y = {0: 4, 1: -2, 2: 3, 3: 1, 4: -3, 5: 1, 6: 5, 7: -3, 8: 0, 9: 5, 10: 4, 11: 0}

        for key in rects:
            tup = rects[key]
            x1 = int(tup[0] + adj_x1[key])
            x2 = int(tup[2] + adj_x2[key])           
            y1 = int(tup[1])
            y2 = int(tup[3])
            cv2.rectangle(new_image, (x1, y1),(x2,y2),(0,255,0),2)

            num_splits = int(abs(y2-y1)//gap)
            for i in range(0, num_splits+1):
                y = int(y1+i*gap) + fine_tune_y[key]
                cv2.line(new_image, (x1, y), (x2, y), (255, 0, 0), 2)
            if key > 0 and key < len(rects) - 1:
                # 竖直线
                x = int((x1+x2) / 2)
                cv2.line(new_image, (x, y1), (x,y2), (0,0, 255), 2)
            
            # 计算数量   除了第一列和最后一列，中间的都是两列的
            if key == 0 or key == len(rects) - 1:
                tot_spots += num_splits + 1
            else:
                tot_spots += 2 * (num_splits + 1)

            # 字典对应好
            if key == 0 or key == len(rects) - 1:
                for i in range(0, num_splits+1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap) + fine_tune_y[key]
                    spot_dict[(x1, y, x2, y+gap)] = cur_len + 1
            else:
                for i in range(0, num_splits+1):
                    cur_len = len(spot_dict)
                    y = int(y1 + i * gap) + fine_tune_y[key]
                    x = int((x1+x2) / 2)
                    spot_dict[(x1, y, x, y+gap)] = cur_len + 1
                    spot_dict[(x, y, x2, y+gap)] = cur_len + 2
        if save:
            filename = 'with_parking.jpg'
            cv2.imwrite(filename, new_image)

        return new_image, spot_dict

new_image, spot_dict = draw_parking(test_image, rects)
cv_imshow('parking_lane_splited', new_image)

# 查看每一个停车位的有效性
tmp_img = test_image.copy()
for k, v in spot_dict.items():
    cv2.rectangle(tmp_img, (int(k[0]),int(k[1])), (int(k[2]),int(k[3])), (0,255,0) , 2)
    cv_imshow(str(v), tmp_img)

# 去掉多余的停车位
invalid_spots = [10, 11, 33, 34, 37, 38, 61, 62, 93, 94, 95, 97, 98, 135, 137, 138, 187, 249, 
           250, 253, 254, 323, 324, 327, 328, 467, 468, 531, 532]
valid_spots_dict = {}
cur_idx = 1
for k, v in spot_dict.items():
    if v in invalid_spots:
        continue
    valid_spots_dict[k] = cur_idx
    cur_idx += 1

print(len(valid_spots_dict))

tmp_img = test_image.copy()
for k, v in valid_spots_dict.items():
    cv2.rectangle(tmp_img, (int(k[0]),int(k[1])), (int(k[2]),int(k[3])), (0,255,0) , 2)
cv_imshow('parking_spot', tmp_img)
cv2.imwrite('valid_pot_img.jpg', tmp_img)
with open('sport_dict.pickle', 'wb') as handle:
    pickle.dump(valid_spots_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#为CNN生成预测图片
def save_images_for_cnn(image, spot_dict, folder_name = "cnn_pred_data"):
    for spot in spot_dict.keys():
        (x1, y1, x2, y2) = spot
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))

        # 裁剪
        spot_img = image[y1:y2, x1:x2]
        spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)
        spot_id = spot_dict[spot]

        filename = 'spot_{}.jpg'.format(str(spot_id))

        print(spot_img.shape, filename, (x1,x2,y1,y2))
        cv2.imwrite(os.path.join(folder_name, filename), spot_img)

save_images_for_cnn(test_image, valid_spots_dict)

