from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2

class ImageClassifier:
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path, compile=False)
        with open(labels_path, "r") as file:
            self.class_names = file.readlines()

    def preprocess_image(self, image):
        """Resize and normalize image for model prediction."""
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1  # Normalize
        return image

    def predict(self, image):
        """Make a prediction and return class name & confidence score."""
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        index = np.argmax(prediction)
        class_name = self.class_names[index]

        # Remove index prefix if class name starts with a number
        class_name = class_name.split(' ', 1)[-1]  # Keeps everything after the first space

        confidence_score = float(prediction[0][index])  # Convert np.float32 to float
        return class_name, confidence_score


# class ImageClassifier:
#     def __init__(self, model_path, labels_path):
#         self.model = load_model(model_path, compile=False)
#         with open(labels_path, "r") as file:
#             self.class_names = file.readlines()
        
#     def preprocess_image(self, image_path):
#         size = (224, 224)
#         image = Image.open(image_path).convert("RGB")
#         image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
#         image_array = np.asarray(image)
#         normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
#         return normalized_image_array.reshape(1, 224, 224, 3)
    
#     def predict(self, image_path):
#         data = self.preprocess_image(image_path)
#         prediction = self.model.predict(data)
#         index = np.argmax(prediction)
#         class_name = self.class_names[index].strip()
#         confidence_score = prediction[0][index]
# #         return class_name, confidence_score

# # Example usage:
# if __name__ == "__main__":
#     model_path = r"C:\Users\User\Intern\RealWear2\model\model_1\keras_model.h5"
#     labels_path = r"C:\Users\User\Intern\RealWear2\model\model_1\labels.txt"
#     image_path = r"C:\Users\User\Intern\RealWear\realWearWebsite\images.jpeg"
    
#     classifier = ImageClassifier(model_path, labels_path)
#     class_name, confidence = classifier.predict(image_path)
    
#     print(f"Class: {class_name}")
#     print(f"Confidence Score: {confidence}")