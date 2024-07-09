import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)

    def capture_image(self):
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to capture image from camera")
        return frame

    def release(self):
        self.camera.release()

if __name__ == "__main__":
    camera = Camera()
    image = camera.capture_image()
    cv2.imshow("Captured Image", image)
    cv2.waitKey(0)
    camera.release()
