import carla

class LaneMarkingType():
    Other = carla.LaneMarkingType.Other
    Broken = carla.LaneMarkingType.Broken
    Solid = carla.LaneMarkingType.Solid
    #// (for double solid line)
    SolidSolid = carla.LaneMarkingType.SolidSolid
    #// (from inside to outside, exception: center lane -from left to right)
    SolidBroken = carla.LaneMarkingType.SolidBroken
    #// (from inside to outside, exception: center lane -from left to right)
    BrokenSolid = carla.LaneMarkingType.BrokenSolid
    #// (from inside to outside, exception: center lane -from left to right)
    BrokenBroken = carla.LaneMarkingType.BrokenBroken
    BottsDots = carla.LaneMarkingType.BottsDots
    #// (meaning a grass edge)
    Grass = carla.LaneMarkingType.Grass
    Curb = carla.LaneMarkingType.Curb
    NONE = carla.LaneMarkingType.NONE
