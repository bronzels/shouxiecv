import torch
import cv2
import os

# 加载YOLOv5模型
model = torch.hub.load('/workspace/shouxiecv/search_on_2d/yolov5-master', 'custom', '/workspace/shouxiecv/search_on_2d/v5pt/yolov5x.pt', source='local')
with open("coco.names", "r") as names_file:
    names = [line.strip() for line in names_file.readlines()]

# 输入图像路径
img_path = 'catperson/2012-03-25_12-57-45_888.jpg'
input_file_name='catperson/2012-03-25_12-57-45_888.jpg'
base, extension = os.path.splitext(input_file_name)
output_file_name = f"{base}-output.{extension}"
input_img = cv2.imread(input_file_name)

# 进行目标检测
results = model(img_path)

# 获取目标数量和检测结果
num_targets = len(results.pred[0]) if results and results.pred is not None else 0

print(f"检测到 {num_targets} 个目标:")

# 遍历每个检测到的目标并打印相似度
for i, det in enumerate(results.pred[0]):
    cls = names[int(det[5])]
    conf = det[4]
    x1, y1, x2, y2 = [int(v) for v in det[0:4]]
    print(f"target {i + 1} - , box:({x1},{y1})({x2},{y2}), conf:{conf:.2f}, cls:{cls}")
    cv2.rectangle(input_img,(x1,y1),(x2,y2),(255,0,255),10)
    cv2.putText(input_img,'cls:{},conf:{:.2f}'.format(cls, conf),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),2)
cv2.imwrite(output_file_name, input_img)
    