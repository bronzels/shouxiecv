import os
import yolov5
import torch
import cv2

# load model
model = yolov5.load('/workspace/shouxiecv/search_on_2d/v5pt/best.pt')
with open("coco.names", "r") as names_file:
    names = [line.strip() for line in names_file.readlines()]

# set image
input_file_name='catperson/2012-03-25_12-57-45_888.jpg'
base, extension = os.path.splitext(input_file_name)
output_file_name = f"{base}-output.{extension}"
input_img = cv2.imread(input_file_name)

# perform inference
#results = model(input_file_name)

# inference with larger input size
results = model(input_file_name, size=1280)

# inference with test time augmentation
#results = model(input_file_name, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, x2, y1, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

if torch.cuda.is_available():
    predictions, boxes, scores, categories = predictions.cpu(), boxes.cpu(), scores.cpu(), categories.cpu()
for (prediction, xyxy, conf, cls) in zip(predictions.numpy(), boxes.numpy(), scores.numpy(), [names[int(cls)] for cls in categories.numpy().tolist()]):
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    #print(f'prediction:{prediction}, box:{box}, score:{score}, category:{category}')
    print(f'box:({x1},{y1})({x2},{y2}), conf:{conf:.2f}, cls:{cls}')
    cv2.rectangle(input_img,(x1,y1),(x2,y2),(255,0,255),10)
    cv2.putText(input_img,'cls:{},conf:{:.2f}'.format(cls, conf),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),2)
cv2.imwrite(output_file_name, input_img)
# show detection bounding boxes on image
#results.show()

# save results into "results/" folder
#results.save(save_dir='catperson/')