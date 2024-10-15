import cv2
import numpy as np
import argparse
import glob
import matplotlib as plt
import torch
import os
from ultralytics.models.yolo import YOLO
from torchvision import models
from torchvision import transforms
from scipy.spatial.distance import euclidean
from PIL import Image

model_emb = models.resnet101(pretrained=False)
pre = torch.load('resnet101-5d3b4d8f.pth')
model_emb.load_state_dict(pre)
model_emb.eval()

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False,
    #default="duoduo-all.jpg",
    default="jess-all.jpg",
    help="path to source image file")
ap.add_argument("-d", "--dest", required=False,
    default="catperson/*.*",
    help="path to destination image folder")
ap.add_argument("-c", "--class", required=False,
    #default="cat",
    default="person",
    help="class of the source image which is known to detect algorithm ")
ap.add_argument("-t", "--confthresh", required=False, type=float,
    default=0.70,
    help="thresh confidence of the detected class")
args = vars(ap.parse_args())

# 读取图片并提取特征
def extract_feature(img, model, transform):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feature = model(tensor)
    return feature.squeeze()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
#img = cv2.imread(args["source"])
img = Image.open(args["source"])
src_feature = extract_feature(img, model_emb, transform)

# 计算两个特征向量的余弦相似度
def cosine_similarity(feature1, feature2):
    return euclidean(feature1, feature2)
    """
    dot_product = torch.sum(feature1 * feature2)
    norm_feature1 = torch.norm(feature1)
    norm_feature2 = torch.norm(feature2)
    similarity = dot_product / (norm_feature1 * norm_feature2)
    return similarity.item()
    """

# 读取图片
src_img = cv2.imread(args["source"])
src_img2detect = src_img

# Load a model
model = YOLO('v8pt/yolo11x.pt')  # load a pretrained model (recommended for training)
# 加载可识别的类名
with open('coco.names', 'r') as f:
    names = f.read().strip().split('\n')

# 为不同的类别创建一个颜色列表
color_list = np.random.uniform(0, 255, size=(len(names), 3))

for input_file_name in glob.glob(args["dest"]):
    base, extension = os.path.splitext(input_file_name)
    base_list = base.split("/")
    output_dir = "/".join(base_list[:-1] + ["out"])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file_name = f"{output_dir}/{base_list[-1]}-{args['class']}{extension}"
    dest_img = cv2.imread(input_file_name)
    dest_img_cpy = dest_img.copy()
    results = model(input_file_name)  # predict on an image
    detected = False
    for result in results:
        for (cls, conf, (xyxy)) in zip([names[int(cls)] for cls in result.boxes.cls.tolist()], result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            print("cls: ", cls)
            print("conf: ", conf)
            x1,y1,x2,y2 = [int(v) for v in xyxy]
            print("xyxy: ", x1,y1,x2,y2)
            if cls != args["class"] or conf < args["confthresh"]:
                continue
            if not detected:
                detected = True
            cropped_img = dest_img_cpy[y1:y2, x1:x2]
            cropped_file_name = f"{output_dir}/{base_list[-1]}-{args['class']}-{x1}_{y1}-{x2}_{y2}{extension}"
            cv2.imwrite(cropped_file_name, cropped_img)
            dest_feature = extract_feature(Image.open(cropped_file_name), model_emb, transform)
            similarity = cosine_similarity(src_feature, dest_feature)
            print("similarity: ", similarity)
            cv2.rectangle(dest_img,(x1,y1),(x2,y2),(255,0,255),10)
            #cv2.putText(dest_img,'cls:{},conf:{:.2f}'.format(cls, conf),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1)
            cv2.putText(dest_img,'similarity:{:.2f}'.format(similarity),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1)
    if detected:
        cv2.imwrite(output_file_name, dest_img)





