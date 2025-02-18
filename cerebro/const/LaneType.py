import carla
# from enum import Enum

class LaneType():
    NONE          = carla.LaneType.NONE
    Driving       = carla.LaneType.Driving
    Stop          = carla.LaneType.Stop
    Shoulder      = carla.LaneType.Shoulder
    Biking        = carla.LaneType.Biking
    Sidewalk      = carla.LaneType.Sidewalk
    Border        = carla.LaneType.Border
    Restricted    = carla.LaneType.Restricted
    Parking       = carla.LaneType.Parking
    Bidirectional = carla.LaneType.Bidirectional
    Median        = carla.LaneType.Median
    Special1      = carla.LaneType.Special1
    Special2      = carla.LaneType.Special2
    Special3      = carla.LaneType.Special3
    RoadWorks     = carla.LaneType.RoadWorks
    Tram          = carla.LaneType.Tram
    Rail          = carla.LaneType.Rail
    Entry         = carla.LaneType.Entry
    Exit          = carla.LaneType.Exit
    OffRamp       = carla.LaneType.OffRamp
    OnRamp        = carla.LaneType.OnRamp
    Any           = carla.LaneType.Any
