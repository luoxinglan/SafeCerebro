import carla

class Location:
    @staticmethod
    def get_Location(x=0.0, y=0.0, z=0):
        return carla.Location(x=x, y=y, z=z)
    
    @staticmethod
    def get_from_forward_vector(forward_vector):
        return carla.Location(forward_vector)