import cv2
from ultralytics import YOLO
import math
import numpy as np
from PIL import Image

class ObjectDetection:
    def __init__(self, model_path, labels_path):
        self.model = YOLO(model_path)
        with open(labels_path, "r") as file:
            self.class_names = [line.strip() for line in file.readlines()]

    # def preprocess_image(self, frame):
    #     if isinstance(frame, str):  # If a file path is given, load the image
    #         frame = cv2.imread(frame)
    #         if frame is None:
    #             raise ValueError(f"Error loading image from path: {frame}")
        
    #     if not isinstance(frame, np.ndarray):
    #         raise ValueError("Invalid input type for frame (expected NumPy array).")

    #     image = cv2.resize(frame, (224, 224))
    #     image = np.asarray(image, dtype=np.float32)
    #     return image
    
    def preprocess_image(self, frame):
        resized_frame = cv2.resize(frame, (640, 640))
        return resized_frame

    def predict(self, frame):
        processed_frame = self.preprocess_image(frame)
        results = self.model(processed_frame, stream=True)

        detected_classes = []
        for r in results:
            if not r.boxes:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                detected_classes.append(class_name)

        return detected_classes


# if __name__ == "__main__":
#     model_path = r"C:\Users\User\Intern\RealWear2\model\objectDetection\yolo11n.pt"
#     labels_path = r"C:\Users\User\Intern\RealWear2\model\objectDetection\yolo_class_names.txt"
#     image_path = r"C:\Users\User\Intern\RealWear2\laptop2.jpg"
    
#     image = cv2.imread(image_path)

#     classifier = ObjectDetection(model_path, labels_path)
#     class_name = classifier.predict(image)
    
#     print(f"Class: {class_name}")