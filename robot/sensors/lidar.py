import numpy as np

class Lidar:
    def __init__(self):
        self.data = np.random.rand(360) * 100  # Simulated LIDAR data

    def get_lidar_data(self):
        return self.data

if __name__ == "__main__":
    lidar = Lidar()
    data = lidar.get_lidar_data()
    print(data)
