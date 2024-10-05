# 导入工具包
import numpy as np
import argparse
import cv2
import myutils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/credit_card_01.png", required=False,
                help="path to input image")
ap.add_argument("-t", "--template", default="ocr_a_reference.png", required=False,
                help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}
# 绘图展示
import matplotlib.pyplot as plt
def cv2_show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
def cv2_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def print_contours_shape(contours):
    shapes = []
    for contour in contours:
        shapes.append(str(contour.shape))
    print(" ".join(shapes))
# 读取一个模板图像
img = cv2.imread(args['template'])
cv2_show(img,"template_img")
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_show(ref,"template_gray")
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)
ref = ref[1]
cv2_show(ref,"template_binary")

# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图），cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓

refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, refCnts, -1, (0,0,255), 3)
cv2_show(img,"template_contours")
print_contours_shape(refCnts)
#排序，从左到右，从上到下
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# 遍历每个轮廓
for (i, c) in enumerate(refCnts):
    # 计算外界矩形并且resize成合适大小
    (x,y,w,h) = cv2.boundingRect(c)
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi, (57, 88))

    # 每个数字对应一个模版
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

#读取输入图像，预处理
image = cv2.imread(args["image"])
cv2_show(image,"image")
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_show(gray,"image_gray")

#礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv2_show(tophat,"image_gray_tophat")
# 
#ksize=-1相当于用3*3
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, 
                  ksize=-1)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
cv2_show(gradX,"image_gray_tophat_gradX")

#通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv2_show(gradX,"image_gray_tophat_gradX_morphclose")
#THRESH_OTSU会寻找适合的阈值，适合双峰，需要阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2_show(thresh,"image_gray_tophat_gradX_morphclose_THRESHOTSU")

#再来一个闭操作

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv2_show(thresh,"image_gray_tophat_gradX_morphclose_THRESHOTSU_morphclose")

# 计算轮廓

threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0,0,255), 3)
cv2_show(cur_img,"image_drawcontours(gray_tophat_gradX_morphclose_THRESHOTSU_morphclose)")
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar > 2.5 and ar < 4.0:

        if (w > 40 and w < 55) and (h > 10 and h < 20):
            #符合的留下来
            locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []

    # 根据坐标提取每一个组
    group = gray[gY - 5:gY +gH + 5, gX - 5:gX +gW + 5]
    cv2_show(group,"image_gray_grouped")
    # 预处理
    group = cv2.threshold(group, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2_show(group,"image_gray_grouped_THRESHOTSU")
    # 计算每一组的轮廓
    digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = myutils.sort_contours(digitCnts,
        method="left-to-right")[0]
    
    # 计算每一个组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))
        cv2_show(roi,"image_gray_grouped_THRESHOTSU_splitBYcontours")

        # 计算匹配得分
        scores = []

        # 在模版中计算每一个得分
        for (digit, digitROI) in digits.items():
            # 模版匹配
            result = cv2.matchTemplate(roi, digitROI,
                cv2.TM_CCOEFF)
            (r1, score, r3, r4) = cv2.minMaxLoc(result)
            scores.append(score)
        
        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))
    
    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5),
                  (gX +gW + 5, gY +gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    
    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2_show(image,"image_group_rectANDtext")






