from ultralytics.models.yolo import YOLO
import torch

# Load a model
#model = YOLO('v8pt/yolo11x.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolo11.yaml', task='detect') 
#11x-batch4

# train the model
model.train(data='./catperson.yaml',
            epochs=200, batch=4, workers=2,
            save=True, imgsz=1024,
            save_period=5, val=True)

