import carla

class VehicleControl():
    @ staticmethod
    def set_control(throttle=0, steer=0, brake=0):
        return carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
