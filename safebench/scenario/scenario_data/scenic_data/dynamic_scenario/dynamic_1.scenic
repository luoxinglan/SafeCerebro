'''The ego vehicle is driving on a straight road when a pedestrian suddenly crosses from the left front and stops as the ego vehicle approaches.'''
Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until (distance from self to EgoTrajectory) < globalParameters.OPT_STOP_DISTANCE
    while True:
        take SetWalkingSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
lane = Uniform(*network.lanes)
EgoTrajectory = lane.centerline
EgoSpawnPt = OrientedPoint on lane.centerline

ego = Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
param OPT_GEO_X_DISTANCE = Range(-8, -2)  # Negative range for left side
param OPT_GEO_Y_DISTANCE = Range(15, 50)

IntSpawnPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = Pedestrian left of IntSpawnPt by globalParameters.OPT_GEO_X_DISTANCE,
    with heading IntSpawnPt.heading - 90 deg,  # Heading perpendicular to the road, adjusted for left crossing
    with regionContainedIn None,
    with behavior AdvBehavior()