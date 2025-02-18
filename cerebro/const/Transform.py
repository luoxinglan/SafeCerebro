import carla

class Transform():

    @ staticmethod
    def get_Transform(location=carla.Location(0,0,0),rotation=carla.Rotation(0,0,0)):
        try:
            return carla.Transform(location,rotation)
        except:
            return carla.Transform()
        