import carla

class Rotation():
    @ staticmethod
    def get_Rotation(roll=0.0, pitch=0.0, yaw=0.0):
        return carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
