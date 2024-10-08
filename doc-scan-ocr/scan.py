import cv2
import matplotlib.pyplot as plt
def cv2_show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
def cv2_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

import numpy as np

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

image = cv2.imread("images/receipt.jpg")
cv2_show(image, 'image')

print(image.shape)

ratio = image.shape[0]/500.0
orig = image.copy()

image = resize(orig, height=500)
cv2_show(image, "image_resized")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 75, 200)

cv2_show(edged, "image_resized_grayed_cannied")

cnts, heirarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4,2), dtype="float32")

    # 下面这个操作，是因为这四个点目前是乱序的，下面通过了一种巧妙的方式来找到对应位置
    
    # 左上和右下， 对于左上的这个点，(x,y)坐标和会最小， 对于右下这个点，(x,y)坐标和会最大，所以坐标求和，然后找最小和最大位置就是了
    # 按照顺序找到对应坐标0123分别是左上， 右上， 右下，左下
    s = pts.sum(axis=1)
    # 拿到左上和右下
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上和左下， 对于右上这个点，(x,y)坐标差会最小，因为x很大，y很小， 而左下这个点， x很小，y很大，所以坐标差会很大
    diff = np.diff(pts, axis=1)
    # 拿到右上和左下
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # 拿到正确的左上，右上， 右下，左下四个坐标点的位置
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算输入的w和h值 这里就是宽度和高度，计算方式就是欧几里得距离，坐标对应位置相减平方和开根号
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    heightB = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置   
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # 计算变换矩阵  变换矩阵这里，要有原始的图像的四个点的坐标， 变换之后的四个点的对应坐标，然后求一个线性矩阵，相当于每个点通过一个线性映射
    # 到新的图片里面去。那么怎么求线性矩阵呢？  
    # 其实解线性方程组， 原始图片四个点坐标矩阵A(4, 2)， 新图片四个点坐标矩阵B(4, 2)， 在列这个维度上扩充1维1
    # A变成了(4, 3), B也是(4, 3)， 每个点相当于(x, y, 1) 
    # B = WA， 其中W是3*3的矩阵，A的每个点是3*1， B的每个点是3*1
    # W矩阵初始化[[h11, h12, h13], [h21, h22, h23], [h31, h32, 1]]  这里面8个未知数，通过上面给出的4个点
    # 所以这里A， B四个点的坐标都扩充了1列，已知A,B四个点的坐标，这里去求参数，解8个线性方程组得到W，就是cv2.getPerspectiveTransform干的事情
    # 这个文章说的不错：https://blog.csdn.net/overflow_1/article/details/80330835
    W = cv2.getPerspectiveTransform(rect, dst)
    # 有了透视矩阵W, 对于原始图片中的每个坐标， 都扩充1列，然后与W乘， 就得到了在变换之后图片的坐标点(x, y, z), 然后把第三列给去掉(x/z, y/z)就是最终的坐标
    warped = cv2.warpPerspective(image, W, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped

warped = four_point_transform(orig, screenCnt.reshape(4,2)*ratio)
print(warped.shape)
warped = cv2.resize(warped, (warped.shape[1]//2*2, warped.shape[0]//2*2))
print(warped.shape)

cv2_show(warped, 'warped')

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY)[1]

cv2_show(ref, 'warped_grayed_thresholded')

rows, cols = ref.shape[:2]
center = (cols/2, rows/2)
angle = 90
scale = 1

M = cv2.getRotationMatrix2D(center, angle, scale)
rotated_img = cv2.warpAffine(ref, M, (cols, rows))

cv2_show(rotated_img, 'warped_rotated')

resized_img = resize(rotated_img, height=600)

cv2_show(resized_img, 'warped_rotated_resized')

cv2.imwrite('images/scan.jpg', resized_img)