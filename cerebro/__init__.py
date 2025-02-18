# mypackage/__init__.py
__version__ = "1.0.0"

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider

SIMULATOR_POLICY_LIST = {
    'CARLA': CarlaDataProvider
}
