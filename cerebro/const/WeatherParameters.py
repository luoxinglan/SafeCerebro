import carla

class WeatherParameters():
    @ staticmethod
    def get_WeatherParameters(sun_altitude_angle=50):
        return carla.WeatherParameters(sun_altitude_angle=sun_altitude_angle)