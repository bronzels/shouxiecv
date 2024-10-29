from ultralytics.models.yolo import YOLO
import cv2
import os
import torch
from PIL import Image
import glob

input_cls = "cat"

# Load a model
'''
model = YOLO('v8pt/yolo11x.pt')  # load a pretrained model (recommended for training)
with open("coco.names", "r") as names_file:
    names = [line.strip() for line in names_file.readlines()]
'''
model = YOLO('runs/detect/train/weights/best.pt')
names = ["duoduo", "jess"]
# Use the model
#results = model.val()  # evaluate model performance on the validation set
#会下载很多测试集文件
output_dir = os.getcwd() + "/catperson-output"
for input_file_name in glob.glob("catperson/*.*"):
#for input_file_name in glob.glob("catperson-test/*.*"):
    #input_file_name='catperson/2012-03-25_12-57-45_888.jpg'
    base, extension = os.path.splitext(input_file_name)
    base_name = base.split("/")[-1]
    output_file_name = f"{output_dir}/{base_name}{extension}"
    input_img = cv2.imread(input_file_name)
    results = model(input_file_name)  # predict on an image
    for result in results:
        for (cls, conf, (xyxy)) in zip([names[int(cls)] for cls in result.boxes.cls.tolist()], result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            print("cls: ", cls)
            print("conf: ", conf)
            x1,y1,x2,y2 = [int(v) for v in xyxy]
            print("xyxy: ", x1,y1,x2,y2)
            cv2.rectangle(input_img,(x1,y1),(x2,y2),(255,0,255),10)
            cv2.putText(input_img,'cls:{},conf:{:.2f}'.format(cls, conf),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),2)
    cv2.imwrite(output_file_name, input_img)