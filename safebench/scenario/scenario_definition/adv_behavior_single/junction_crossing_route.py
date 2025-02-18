''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-30 21:56:48
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla
import numpy as np
from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms


class OppositeVehicleRunningRedLight(BasicScenario):
    """
        This class holds everything required for a scenario, in which an other vehicle takes priority from the ego vehicle, 
        by running a red traffic light (while the ego vehicle has green).
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight-Behavior-Single", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location of the ego vehicle found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        base_speed = 5.0
        speed_scale = 5.0
        # speed = actions[0] * speed_scale + base_speed
        try:
            speed = actions[0] * speed_scale + base_speed
        except:
            speed = 1.0*np.random.rand()* speed_scale + base_speed #caixuan
        return speed

    def initialize_actors(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.other_actor_delta_x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
            other_actor_transform.rotation
        )

        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location of the other vehicle found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Red)
            traffic_light_other.set_red_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] initial action.'
        self.other_actor_delta_x = 1.0
        self.trigger_distance_threshold = 35

    def update_behavior(self, scenario_action):
        other_actor_speed = self.convert_actions(scenario_action)
        for i in range(len(self.other_actors)):
            self.scenario_operation.go_straight(other_actor_speed, i)

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionLeftTurn(BasicScenario):
    """
        Vehicle turning left at signalized junction scenario. 
        An actor has higher priority, ego needs to yield to oncoming actor.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionLeftTurn, self).__init__("SignalizedJunctionLeftTurn-Behavior-Single", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

        # other vehicle's traffic light
        self.scenario_operation = ScenarioOperation()
        self.reference_actor = None
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        base_speed = 1.0
        speed_scale = 20.0
        # speed = actions[0] * speed_scale + base_speed
        try:
            speed = actions[0] * speed_scale + base_speed
        except:
            speed = 1*np.random.rand()* speed_scale + base_speed #caixuan
        return speed

    def initialize_actors(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.other_actor_delta_x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(
                other_actor_transform.location.x, 
                other_actor_transform.location.y, 
                other_actor_transform.location.z
            ),
            other_actor_transform.rotation
        )
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Green)
            traffic_light_other.set_green_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] initial action.'
        self.other_actor_delta_x = 1.0
        self.trigger_distance_threshold = 60

    def update_behavior(self, scenario_action):
        other_actor_speed = self.convert_actions(scenario_action)
        for i in range(len(self.other_actors)):
            self.scenario_operation.go_straight(other_actor_speed, i)

    def check_stop_condition(self):
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionRightTurn(BasicScenario):
    """
        Vehicle turning right at signalized junction scenario an actor has higher priority, ego needs to yield to oncoming actor
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(SignalizedJunctionRightTurn, self).__init__("SignalizedJunctionRightTurn-Behavior-Single", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Red)
            self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger = False
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        base_speed = 5.0
        speed_scale = 5.0
        # speed = actions[0] * speed_scale + base_speed
        try:
            speed = actions[0] * speed_scale + base_speed
        except:
            speed = 0.8*np.random.rand()* speed_scale + base_speed #caixuan
        return speed

    def initialize_actors(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.other_actor_delta_x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z),
            other_actor_transform.rotation
        )
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(other_actor_transform, False, True)
        if traffic_light_other is None:
            print(">> No traffic light for the given location found")
        else:
            traffic_light_other.set_state(carla.TrafficLightState.Green)
            traffic_light_other.set_green_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] initial action.'
        self.other_actor_delta_x = 1.0
        self.trigger_distance_threshold = 45

    def update_behavior(self, scenario_action):
        other_actor_speed = self.convert_actions(scenario_action)
        for i in range(len(self.other_actors)):
            self.scenario_operation.go_straight(other_actor_speed, i)

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class NoSignalJunctionCrossingRoute(BasicScenario):
    """
        Vehicle turning right at an intersection without traffic lights.
    """
    
    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(NoSignalJunctionCrossingRoute, self).__init__("NoSignalJunctionCrossingRoute-Behavior-Single", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self.scenario_operation = ScenarioOperation()
        self.reference_actor = None
        
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150
        self.ran_cof = np.random.rand() #caixuan，只在每次场景重启时进行随机化

    def convert_actions(self, actions):
        base_speed = 9.0 # 9
        speed_scale = 3.0 # PROPOSED：3
        try:
            speed = actions[0] * speed_scale + base_speed
        except:
            speed = 1*self.ran_cof* speed_scale + base_speed #caixuan
        return speed

    def initialize_actors(self):
        other_actor_transform = self.config.other_actors[0].transform
        forward_vector = other_actor_transform.rotation.get_forward_vector() * self.other_actor_delta_x
        other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(
                other_actor_transform.location.x, 
                other_actor_transform.location.y, 
                other_actor_transform.location.z
            ),
            other_actor_transform.rotation
        )
        self.actor_transform_list = [first_vehicle_transform]
        self.actor_type_list = ["vehicle.audi.tt"]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
    
    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] initial action.'
        self.other_actor_delta_x = 1.0
        self.trigger_distance_threshold = 35

    def update_behavior(self, scenario_action):
        other_actor_speed = self.convert_actions(scenario_action)
        for i in range(len(self.other_actors)):
            self.scenario_operation.go_straight(other_actor_speed, i)

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False
