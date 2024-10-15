from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from label_studio_ml.utils import get_single_tag_keys, get_local_path
import requests, os
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
 
LS_URL = os.environ['LABEL_STUDIO_URL']
LS_API_TOKEN = os.environ['LABEL_STUDIO_API_KEY']
LS_MODEL_PATH = os.environ['LABEL_STUDIO_MODEL_PATH']
LS_MODEL_CONF = float(os.environ['LABEL_STUDIO_MODEL_CONF'])
LS_MODEL_VERSION = os.environ['LABEL_STUDIO_MODEL_VERSION']

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")
        
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        self.model = YOLO(LS_MODEL_PATH)
        self.labels = self.model.names

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]
        
        #return ModelResponse(predictions=[])
        
        task = tasks[0]
        url = tasks[0]['data']['image']
        print(f'url is: {url}')
        image_path = self.get_local_path(url=url,ls_host=LS_URL,task_id=tasks[0]['id'])
        print(f'image_path: {image_path}')
        image = Image.open(image_path)
        original_width, original_height = image.size

        predictions = []
        score = 0
        i = 0

        results = self.model.predict(image,conf=LS_MODEL_CONF)        

        for result in results:
            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                index = int(prediction.cls[0].tolist())
                print("index: ", index)
                cls = self.labels[index]
                predictions.append({
                    #"id": str(i),
                    "id": cls,
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "score": prediction.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": xyxy[0] / original_width * 100, 
                        "y": xyxy[1] / original_height * 100,
                        "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                        "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                        "rectanglelabels": [self.labels[int(prediction.cls.item())]]
                    }
                })
                score += prediction.conf.item()
        print(f"Prediction Score is {score:.3f}.")    
        final_prediction = [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": LS_MODEL_VERSION 
        }]
        return ModelResponse(predictions=final_prediction)
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

