#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


cap = cv2.VideoCapture('02_Video/01_Foreground.avi')

# 角点检测所需参数
# 如果不限制角点数量，速度就会有些慢，达不到实时的效果
# 品质因子会筛选角点，品质因子设置的越大，得到的角点越少
'''
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7 )
'''
feature_params = dict( maxCorners = 30,
                       qualityLevel = 0.5,
                       minDistance = 15 )

# lucas-kanada参数
# winSize：窗口大小 maxLevel：金字塔层数
lk_params = dict( winSize = (15,15),
                  maxLevel = 2)

# 随机颜色条
color = np.random.randint(0,255,(1000,3))

# 拿到第一帧
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# cv2.goodFeaturesToTrack函数返回所有检测特征点，需要输入：图像，角点最大数量（效率），品质因子（特征值越大的越好来筛选）
# 距离相当于这区间有比这个角点强的，就不要这个弱的了
# **变量 作为传入参数，是用来传入不定长的变量
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# 创建一个mask
mask = np.zeros_like(old_frame)

while(True):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # st=1 表示找到的特征点，没找到的特征点就不要了
    good_new = p1[st==1]
    good_old = p0[st==1]

    for i, (new,old) in enumerate(zip(good_new,good_old)):
        a,b = (int(ele) for ele in new.ravel())
        c,d = (int(ele) for ele in old.ravel())

        mask = cv2.line(mask,(a,b),(c,d),color[i].tolist(),2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

    # 更新
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    p0 = np.round(p0)
    p0_new = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    p0_merged = np.vstack((p0,p0_new))
    p0_merged = p0_merged.reshape((-1,2))
    p0_merged_deduped = np.unique(p0_merged,axis=0)
    p0_merged_deduped = p0_merged_deduped.reshape(-1,1,2)
    print(p0_merged_deduped.shape[0] - p0.shape[0])
    p0 = p0_merged_deduped

cv2.destroyAllWindows()
cap.release()



# 
