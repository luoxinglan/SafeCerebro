# from enum import Enum
from cerebro.const.LaneType import LaneType
from cerebro.const.Location import Location
from cerebro.const.LaneMarkingColor import LaneMarkingColor
from cerebro.const.LaneMarkingType import LaneMarkingType
from cerebro.const.Transform import Transform
from cerebro.const.VehicleControl import VehicleControl
from cerebro.const.TrafficLight import TrafficLight
from cerebro.const.libcarla import libcarla
from cerebro.const.WeatherParameters import WeatherParameters
from cerebro.const.Vehicle import Vehicle
from cerebro.const.Rotation import Rotation

simulator_type = 'CARLA'
    
class cerebro():
    LaneType = LaneType
    Location = Location
    LaneMarkingColor = LaneMarkingColor
    LaneMarkingType = LaneMarkingType
    Transform = Transform
    VehicleControl = VehicleControl
    TrafficLight = TrafficLight
    libcarla = libcarla
    WeatherParameters = WeatherParameters
    Vehicle = Vehicle
    Rotation = Rotation