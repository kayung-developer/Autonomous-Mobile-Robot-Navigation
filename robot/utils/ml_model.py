import cv2
import numpy as np
import tensorflow as tf

class MLModel:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        # Load pre-trained ML model (e.g., TensorFlow, PyTorch)
        return tf.keras.models.load_model('path/to/your/model')

    def detect_obstacles(self, image):
        # Pre-process the image
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict obstacles
        predictions = self.model.predict(image)
        return predictions

if __name__ == "__main__":
    ml_model = MLModel()
    image = cv2.imread('path/to/sample/image.jpg')
    obstacles = ml_model.detect_obstacles(image)
    print(obstacles)
