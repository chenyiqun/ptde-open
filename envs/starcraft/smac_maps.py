from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib
from smac.env.starcraft2.maps import smac_maps

map_param_registry = {
    "1o_10b_vs_1r": {
        "n_agents": 11,
        "n_enemies": 1,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_bane"
    },
    "1o_2r_vs_4r": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_roach"
    },
        "bane_vs_hM": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    },
        "3s_vs_6z": {
        "n_agents": 3,
        "n_enemies": 6,
        "limit": 300,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers"
    },
        "3s_vs_8z": {
        "n_agents": 3,
        "n_enemies": 8,
        "limit": 350,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers"
    },
        "corridor2": {
        "n_agents": 4,
        "n_enemies": 24,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
        "3m_vs_4m": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 50,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
        "6h_vs_9z": {
        "n_agents": 6,
        "n_enemies": 9,
        "limit": 200,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "hydralisks",
    },
        "MMM3": {
        "n_agents": 10,
        "n_enemies": 13,
        "limit": 80,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
        "MMM4": {
        "n_agents": 10,
        "n_enemies": 13,
        "limit": 80,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
        "MMM5": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 80,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
        "3s5z_vs_4s5z": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
        "3s5z_vs_3s7z": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    }
}


smac_maps.map_param_registry.update(map_param_registry)

def get_map_params(map_name):
    map_param_registry = smac_maps.get_smac_map_registry()
    return map_param_registry[map_name]


for name in map_param_registry.keys():
    globals()[name] = type(name, (smac_maps.SMACMap,), dict(filename=name))
