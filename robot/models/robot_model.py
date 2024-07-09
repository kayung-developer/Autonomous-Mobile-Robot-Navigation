class RobotModel:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation

    def update_position(self, new_position):
        self.position = new_position

    def update_orientation(self, new_orientation):
        self.orientation = new_orientation

    def get_state(self):
        return self.position, self.orientation
