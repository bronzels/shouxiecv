import cv2
import numpy as np
import argparse
import glob
import matplotlib as plt
from util import detect_outliers
from sklearn.cluster import KMeans

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False,
    #default="duoduo-head-edge.jpg",
    #default="jess.jpg",
    default="book.jpg",
    help="path to source image file")
ap.add_argument("-d", "--dest", required=False,
    default="book/*.*",
    help="path to destination image folder")
#surf提示需要重新编译
ap.add_argument("-a", "--algorithm", required=False,
    default="orb",
    help="algorithm to detect")
ap.add_argument("-m", "--matcher", required=False,
    default="bf",
    help="matcher to use")
ap.add_argument("-k", "--knn", required=False, type=bool,
    default=False,
    help="use knn or not")
args = vars(ap.parse_args())

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 读取图片
src_img = cv2.imread(args["source"])
src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
#cv_show('src_img_gray', src_img_gray)
#_, src_img_thresh = cv2.threshold(src_img_gray, 120, 255, cv2.THRESH_BINARY)
#cv_show('src_img_thresh', src_img_thresh)
src_img_blur = cv2.GaussianBlur(src_img_gray, (3,3), 0)
src_img_edge = cv2.Canny(src_img_blur, 75, 200)
#contours, _ = cv2.findContours(src_img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 在原图上绘制轮廓
#cv2.drawContours(src_img, contours, -1, (0, 255, 0), 2)
#cv_show('src_img', src_img)
src_img2detect = src_img_gray
#src_img2detect = src_img_edge
cv_show('src_img2detect', src_img2detect)

dst_images = [cv2.imread(path) for path in glob.glob(args["dest"])]
dst_images_gray = [ cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in dst_images]
dst_images_blur = [ cv2.GaussianBlur(img, (3,3), 0) for img in dst_images_gray]
dst_images_edge = [ cv2.Canny(img, 75, 200) for img in dst_images_gray]

# 初始化检测器
if args["algorithm"] == "orb":
    # ORB
    detector = cv2.ORB_create()
    dist_algo = cv2.NORM_HAMMING
elif args["algorithm"] == "sift":
    # SIFT
    detector = cv2.xfeatures2d.SIFT_create()
    dist_algo = cv2.NORM_L2
elif args["algorithm"] == "surf":
    # SURF
    detector = cv2.xfeatures2d.SURF_create()
    dist_algo = cv2.NORM_L2
else:
    pass
# 创建matcher对象
if args["knn"]:
    if args["matcher"] == "bf":
        matcher = cv2.BFMatcher(dist_algo)
    else:
        matcher = cv2.FlannBasedMatcher(dist_algo)
else:
    if args["matcher"] == "bf":
        matcher = cv2.BFMatcher(dist_algo, crossCheck = True)
    else:
        matcher = cv2.FlannBasedMatcher(dist_algo, crossCheck = True)
kp_src,des_src = detector.detectAndCompute(src_img2detect, None)
# 打印关键点数量
print(f'Detected {len(kp_src)} keypoints of src image')

"""
for i in range(len(dst_images)):
    dst_img = dst_images[i]
    dst_img_gray = dst_images_gray[i]
    cascade_detector = cv2.CascadeClassifier("/Volumes/data/envs/opencv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalcatface.xml")
    #haarcascade_frontalcatface_extended.xml
    rects = cascade_detector.detectMultiScale(dst_img_gray, scaleFactor=1.1,
    minNeighbors=10, minSize=(100, 100))
    # loop over the cat faces and draw a rectangle surrounding each
    print (enumerate(rects))
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(dst_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(dst_img, "Cat #{}".format(i + 1), (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        print (i, x,y,w,h)
        # show the detected cat faces
        cv2.imshow("Cat Faces", dst_img)
        cv2.waitKey()
"""

for i in range(len(dst_images)):
    dst_img = dst_images[i]
    dst_img_gray = dst_images_gray[i]
    dst_img_edge = dst_images_edge[i]
    dst_img2detect = dst_img_gray
    #dst_img2detect = dst_img_edge
    cv_show('dst_img2detect', dst_img2detect)

    kp_dst,des_dst = detector.detectAndCompute(dst_img2detect, None)

    def filter_outliers(kp, des):
        kmeans = KMeans(n_clusters=1)
        points = np.int32([[kp[dp.trainIdx].pt[0], kp[dp.trainIdx].pt[1]] for dp in des])
        kmeans.fit(points)
        labels, mask = detect_outliers(points, kmeans)
        print("labels:", labels)
        return np.delete(des, labels)
        #return des[mask]

    # 匹配描述符
    if args["knn"]:
        matches = matcher.knnMatch(des_src, des_dst, k=2)
        # 筛选比较好的匹配点
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
            #if m.distance < 0.65 * n.distance:
            #if m.distance < 0.55 * n.distance:
            #if m.distance < 0.45 * n.distance:
                good.append([m])
        good = filter_outliers(kp_dst, good)
        img_matches = cv2.drawMatchesKnn(src_img, kp_src, dst_img, kp_dst, good, None, flags=2)
    else:
        matches = matcher.match(des_src, des_dst)
        matches = sorted(matches, key = lambda x: x.distance)
        good = matches[:20]
        # 按照距离排序
        # 绘制前n个匹配项
        good = filter_outliers(kp_dst, good)
        img_matches = cv2.drawMatches(src_img, kp_src, dst_img, kp_dst, good, None, flags=2)

    cv_show("img_matches", img_matches)



