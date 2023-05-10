# Convert the hamlet scenario format to lemlab scenario format

import pandas as pd
import os
import shutil
import time
import json
from ruamel.yaml import YAML
from pprint import pprint
import ast
import numpy as np
import string
from random import choice

INPUT_PATH = os.path.join('.', '04 - scenarios', 'example_small')
OUTPUT_PATH = os.path.join('.', 'scenarios')
FCAST_RETRAINING_PERIOD = 86400
FCAST_UPDATE_PERIOD = 900


class h2l:

    def __init__(self, input_path: str = None, output_path: str = None, name: str = None,
                 market: str = 'lem_continuous'):
        self.input_path = input_path if input_path is not None else INPUT_PATH
        self.name = name if name is not None else self.input_path.rsplit("\\", 1)[-1]
        self.output_path = output_path if output_path is not None else os.path.join(OUTPUT_PATH, self.name)
        self.market = market
        print(f"Converting hamlet scenario from '{self.input_path}' to lemlab scenario '{self.output_path}'")

        self.naming = {
            # Config is structured the following way:
            # 1. level: source files and parameters
            # 2. level: parameter categories in lemlab
            # 3. level: parameter subcategories in lemlab
            # 4. level: parameter names in lemlab
            # 5. level: parameter values in lemlab (either set value or a list with set of instructions
            #           [source file index, path to find it in HAMLET, function to call]
            'config': {
                'sources': ['config/config_general.yaml', 'config/config_markets.yaml'],
                'params': {
                    'simulation': {
                        'rts': False,
                        'lem_active': True,
                        'agents_active': True,
                        'rts_start_steps': 6,
                        'sim_start': [0, ['simulation', 'sim', 'start'], self.__config_sim_start],
                        'sim_start_tz': 'europe/berlin',
                        'sim_length': [0, ['simulation', 'sim', 'duration'], self.__get_value],
                        'path_input_data': '../input_data',
                        'path_scenarios': '../scenarios',
                    },
                    'lem': {
                        'types_clearing_ex_ante': {0: 'pda'},
                        'types_clearing_ex_post': {0: 'community'},
                        'types_pricing_ex_ante': {0: 'uniform',
                                                  1: 'discriminatory'},
                        'types_pricing_ex_post': {0: 'standard'},
                        'share_quality_logging_extended': True,
                        'types_quality': {0: 'na',
                                          1: 'local',
                                          2: 'green_local'},
                        'types_position': {0: 'offer',
                                           1: 'bid'},
                        'types_transaction': {0: 'market',
                                              1: 'balancing',
                                              2: 'levy_prices'},
                        'positions_delete': True,
                        'positions_archive': True,
                        'horizon_clearing': 86400,
                        'interval_clearing': 900,
                        'frequency_clearing': 900,
                        'calculate_virtual_submeters': True,
                        'prices_settlement_in_advance': 0,
                        'types_meter': {0: 'plant submeter',
                                        1: 'virtual plant submeter',
                                        2: 'dividing meter',
                                        3: 'virtual dividing meter',
                                        4: 'grid meter',
                                        5: 'virtual grid meter'},
                        'bal_energy_pricing_mechanism': [1, [self.market, 'pricing', 'retailer', 'balancing', 'method'],
                                                         self.__get_value],
                        'path_bal_prices': [1, [self.market, 'pricing', 'retailer', 'balancing', 'file', 'file'],
                                            self.__get_value],
                        'price_energy_balancing_positive': [1,
                                                            [self.market, 'pricing', 'retailer', 'balancing', 'fixed',
                                                             'price'], [0], self.__get_value_list],
                        'price_energy_balancing_negative': [1,
                                                            [self.market, 'pricing', 'retailer', 'balancing', 'fixed',
                                                             'price'], [1], self.__get_value_list],
                        'levy_pricing_mechanism': [1, [self.market, 'pricing', 'retailer', 'levies', 'method'],
                                                   self.__get_value],
                        'path_levy_prices': [1, [self.market, 'pricing', 'retailer', 'levies', 'file', 'file'],
                                             self.__get_value],
                        'price_energy_levies_positive': [1, [self.market, 'pricing', 'retailer', 'levies', 'fixed',
                                                             'price'], [0], self.__get_value_list],
                        'price_energy_levies_negative': [1, [self.market, 'pricing', 'retailer', 'levies', 'fixed',
                                                             'price'], [1], self.__get_value_list],
                    },
                    'retailer': {
                        'retail_pricing_mechanism': [1, [self.market, 'pricing', 'retailer', 'energy', 'method'],
                                                     self.__get_value],
                        'path_retail_prices': [1, [self.market, 'pricing', 'retailer', 'energy', 'file', 'file'],
                                               self.__get_value],
                        'price_sell': [1, [self.market, 'pricing', 'retailer', 'energy', 'fixed', 'price'], [1],
                                       self.__get_value_list],
                        'price_buy': [1, [self.market, 'pricing', 'retailer', 'energy', 'fixed', 'price'], [0],
                                      self.__get_value_list],
                        'qty_energy_bid': [1, [self.market, 'pricing', 'retailer', 'energy', 'fixed', 'quantity'], [1],
                                           self.__get_value_list],
                        'qty_energy_offer': [1, [self.market, 'pricing', 'retailer', 'energy', 'fixed', 'quantity'],
                                             [0], self.__get_value_list],
                        'quality': 'na',
                        'id_user': 'retailer01',
                    },
                    'aggregator': {
                        'active': False,
                    },
                    'db_connections': {
                        'database_connection_admin': {
                            'user': [0, ['database', 'admin', 'user'], self.__get_value],
                            'pw': [0, ['database', 'admin', 'pw'], self.__get_value],
                            'host': [0, ['database', 'admin', 'host'], self.__get_value],
                            'port': [0, ['database', 'admin', 'port'], self.__get_value],
                            'db': [0, ['database', 'admin', 'db'], self.__get_value],
                        },
                        'database_connection_user': {
                            'user': [0, ['database', 'user', 'user'], self.__get_value],
                            'pw': [0, ['database', 'user', 'pw'], self.__get_value],
                            'host': [0, ['database', 'user', 'host'], self.__get_value],
                            'port': [0, ['database', 'user', 'port'], self.__get_value],
                            'db': [0, ['database', 'user', 'db'], self.__get_value],
                        },
                    },
                },
            },
            # Agent is structured the following way:
            # 1. level: agent type (e.g. 'sfh', 'mfh')
            # 2. level: agent files in HAMLET
            'agents': {
                'type': ['sfh'],
                'sfh': {
                    # 3. level: agent parameter categories in HAMLET
                    # 4. level: agent parameter names in lemlab
                    # 5. level: instructions on how to convert HAMLET to lemlab values (name: name in HAMLET, type: type
                    #   of value, func: function to convert value but can also be used as value if no name is provided)
                    'account.json': {
                        'general': {
                            'id_user': {
                                'src': 'agent_id',
                                'type': str,
                                'func': self._conv_idx,
                            },
                            'id_market_agent': {
                                'src': 'agent_id',
                                'type': str,
                                'func': self._conv_idx,
                            },
                            'id_user_old': {
                                'src': 'agent_id',
                                'type': str,
                                'func': None,
                            },
                            'solver': {
                                'src': None,
                                'type': str,
                                'func': 'gurobi',
                            },
                        },
                        'mpc': {
                            'mpc_horizon': {
                                'src': 'horizon',
                                'type': int,
                                'func': None,
                            },
                            'mpc_price_fcast': {
                                'src': 'price_fcast',
                                'type': str,
                                'func': None,
                            },
                            'mpc_price_fcast_retraining_period': {
                                'src': None,
                                'type': str,
                                'func': 86400,
                            },
                            'mpc_price_fcast_update_period': {
                                'src': None,
                                'type': str,
                                'func': 900,
                            },
                            'controller_strategy': {
                                'src': None,
                                'type': str,
                                'func': 'mpc_opt',
                            },
                        },
                        'market_agent': {
                            'ma_horizon': {
                                'src': 'horizon',
                                'type': int,
                                'func': None,
                            },
                            'ma_strategy': {
                                'src': 'strategy',
                                'type': str,
                                'func': None,
                            },
                            'ma_preference_quality': {
                                'src': 'preference_quality',
                                'type': str,
                                'func': None,
                            },
                            'ma_premium_preference_quality': {
                                'src': 'premium_preference_quality',
                                'type': int,
                                'func': None,
                            },
                        },
                        'meter': {
                            'meter_prob_late': {
                                'src': 'prob_late',
                                'type': int,
                                'func': None,
                            },
                            'meter_prob_late_95': {
                                'src': 'prob_late_95',
                                'type': int,
                                'func': None,
                            },
                            'meter_prob_missing': {
                                'src': 'prob_missing',
                                'type': int,
                                'func': None,
                            },
                        },
                        'plants': {
                            'list_plants': {
                                'src': 'plants',
                                'type': None,
                                'func': None,
                            },
                            'id_meter_grid': {
                                'src': None,
                                'type': str,
                                'func': self._create_main_meter,
                            },
                        },
                    },
                    'plants.json': {
                        'hh': {
                            'src': 'inflexible_load',
                            'params': {
                                'type': {
                                    'src': None,
                                    'type': str,
                                    'func': 'hh',
                                },
                                'activated': {
                                    'src': None,
                                    'type': None,
                                    'func': True,
                                },
                                'has_submeter': {
                                    'src': None,
                                    'type': None,
                                    'func': True,
                                },
                                'fcast': {
                                    'src': ['fcast', 'method'],
                                    'type': str,
                                    'func': None,
                                },
                                'fcast_order': {
                                    'src': ['fcast', 'sarma_order'],
                                    'type': None,
                                    'func': self._conv_str_to_list,
                                },
                                'fcast_retraining_period': {
                                    'src': None,
                                    'type': int,
                                    'func': FCAST_RETRAINING_PERIOD,
                                },
                                'fcast_update_period': {
                                    'src': None,
                                    'type': int,
                                    'func': FCAST_UPDATE_PERIOD,
                                },
                                'annual_consumption': {
                                    'src': 'demand',
                                    'type': int,
                                    'func': self._divide_by_1000,
                                },
                            },
                        },
                        'pv': {
                            'src': 'pv',
                            'params': {
                                'type': {
                                    'src': 'type',
                                    'type': str,
                                    'func': None,
                                },
                                'activated': {
                                    'src': None,
                                    'type': None,
                                    'func': True,
                                },
                                'has_submeter': {
                                    'src': None,
                                    'type': None,
                                    'func': True,
                                },
                                'power': {
                                    'src': 'power',
                                    'type': int,
                                    'func': None,
                                },
                                'controllable': {
                                    'src': 'controllable',
                                    'type': bool,
                                    'func': None,
                                },
                                'fcast': {
                                    'src': ['fcast', 'method'],
                                    'type': str,
                                    'func': None,
                                },
                                'fcast_order': {
                                    'src': None,
                                    'type': None,
                                    'func': [],
                                },
                                'fcast_param': {
                                    'src': ['fcast', 'smoothed_timesteps'],
                                    'type': None,
                                    'func': None,
                                },
                                'fcast_retraining_period': {
                                    'src': None,
                                    'type': int,
                                    'func': FCAST_RETRAINING_PERIOD,
                                },
                                'fcast_update_period': {
                                    'src': None,
                                    'type': int,
                                    'func': FCAST_UPDATE_PERIOD,
                                },
                                'quality': {
                                    'src': 'quality',
                                    'type': str,
                                    'func': None,
                                },
                            },
                        },
                        'ev': {
                            'src': 'ev',
                            'params': {
                                'type': {
                                    'src': 'type',
                                    'type': str,
                                    'func': None,
                                },
                                'activated': {
                                    'src': None,
                                    'type': bool,
                                    'func': True,
                                },
                                'has_submeter': {
                                    'src': None,
                                    'type': bool,
                                    'func': True,
                                },
                                'efficiency': {
                                    'src': 'charging_efficiency',
                                    'type': float,
                                    'func': None,
                                },
                                'v2g': {
                                    'src': 'v2g',
                                    'type': bool,
                                    'func': None,
                                },
                                'charging_power': {
                                    'src': 'charging_home',
                                    'type': int,
                                    'func': None,
                                },
                                'capacity': {
                                    'src': 'capacity',
                                    'type': int,
                                    'func': None,
                                },
                                'consumption': {
                                    'src': 'consumption',
                                    'type': int,
                                    'func': None,
                                },
                                'fcast': {
                                    'src': ['fcast', 'method'],
                                    'type': str,
                                    'func': None,
                                },
                                'fcast_order': {
                                    'src': None,
                                    'type': None,
                                    'func': [],
                                },
                                'fcast_param': {
                                    'src': None,
                                    'type': None,
                                    'func': [],
                                },
                                'fcast_retraining_period': {
                                    'src': None,
                                    'type': int,
                                    'func': FCAST_RETRAINING_PERIOD,
                                },
                                'fcast_update_period': {
                                    'src': None,
                                    'type': int,
                                    'func': FCAST_UPDATE_PERIOD,
                                },
                                'quality': {
                                    'src': ['fcast', 'quality'],
                                    'type': str,
                                    'func': None,
                                },
                            },
                        },
                        'bat': {
                            'src': 'battery',
                            'params': {
                                'type': {
                                    'src': None,
                                    'type': str,
                                    'func': 'bat',
                                },
                                'activated': {
                                    'src': None,
                                    'type': bool,
                                    'func': True,
                                },
                                'has_submeter': {
                                    'src': None,
                                    'type': bool,
                                    'func': True,
                                },
                                'power': {
                                    'src': 'power',
                                    'type': int,
                                    'func': None,
                                },
                                'capacity': {
                                    'src': 'capacity',
                                    'type': int,
                                    'func': None,
                                },
                                'efficiency': {
                                    'src': 'efficiency',
                                    'type': float,
                                    'func': None,
                                },
                                'charge_from_grid': {
                                    'src': 'g2b',
                                    'type': bool,
                                    'func': None,
                                },
                                'quality': {
                                    'src': 'quality',
                                    'type': str,
                                    'func': None,
                                },
                            },
                        },
                        # 'hp': {
                        #     'src': 'hp',
                        #     'params': {
                        #         'type': {
                        #             'src': 'type',
                        #             'type': str,
                        #             'func': None,
                        #         },
                        #         'activated': {
                        #             'src': None,
                        #             'type': bool,
                        #             'func': True,
                        #         },
                        #         'has_submeter': {
                        #             'src': None,
                        #             'type': bool,
                        #             'func': True,
                        #         },
                        #         'power_th': {
                        #             'src': 'power',
                        #             'type': int,
                        #             'func': None,
                        #         },
                        #         'hp_type': {
                        #             'src': None,
                        #             'type': str,
                        #             'func': 'Outdoor Air/Water',
                        #         },
                        #         'temperature': {
                        #             'src': None,
                        #             'type': int,
                        #             'func': 50,
                        #         },
                        #         'capacity': {
                        #             'src': None,
                        #             'type': int,
                        #             'func': self._get_storage_value,
                        #         },
                        #         'efficiency': {
                        #             'src': None,
                        #             'type': float,
                        #             'func': self._get_storage_value,
                        #         },
                        #         'fcast': {
                        #             'src': ['fcast', 'method'],
                        #             'type': str,
                        #             'func': None,
                        #         },
                        #         'fcast_order': {
                        #             'src': None,
                        #             'type': None,
                        #             'func': [],
                        #         },
                        #         'fcast_param': {
                        #             'src': ['fcast', 'smoothed_timesteps'],
                        #             'type': None,
                        #             'func': None,
                        #         },
                        #         'fcast_retraining_period': {
                        #             'src': None,
                        #             'type': int,
                        #             'func': FCAST_RETRAINING_PERIOD,
                        #         },
                        #         'fcast_update_period': {
                        #             'src': None,
                        #             'type': int,
                        #             'func': FCAST_UPDATE_PERIOD,
                        #         },
                        #     },
                        # },
                    },
                    # 3. level: call function to convert each column into a separate meter file
                    'meters.ft': self._split_meter,
                    'socs.ft': self._split_soc,
                    'timeseries.ft': self._split_timeseries,
                },
            },
            # Market is structured the following way (check that identical with config):
            # 1. level: source files and parameters
            # 2. level: parameter names in lemlab
            'market': {
                'sources': ['config/config_markets.yaml'],
                'params': {
                        'types_clearing_ex_ante': {0: 'pda'},
                        'types_clearing_ex_post': {0: 'community'},
                        'types_pricing_ex_ante': {0: 'uniform',
                                                  1: 'discriminatory'},
                        'types_pricing_ex_post': {0: 'standard'},
                        'share_quality_logging_extended': True,
                        'types_quality': {0: 'na',
                                          1: 'local',
                                          2: 'green_local'},
                        'types_position': {0: 'offer',
                                           1: 'bid'},
                        'types_transaction': {0: 'market',
                                              1: 'balancing',
                                              2: 'levy_prices'},
                        'positions_delete': True,
                        'positions_archive': True,
                        'horizon_clearing': 86400,
                        'interval_clearing': 900,
                        'frequency_clearing': 900,
                        'calculate_virtual_submeters': True,
                        'prices_settlement_in_advance': 0,
                        'types_meter': {0: 'plant submeter',
                                        1: 'virtual plant submeter',
                                        2: 'dividing meter',
                                        3: 'virtual dividing meter',
                                        4: 'grid meter',
                                        5: 'virtual grid meter'},
                        'bal_energy_pricing_mechanism': [0, [self.market, 'pricing', 'retailer', 'balancing', 'method'],
                                                         self.__get_value],
                        'path_bal_prices': [0, [self.market, 'pricing', 'retailer', 'balancing', 'file', 'file'],
                                            self.__get_value],
                        'price_energy_balancing_positive': [0,
                                                            [self.market, 'pricing', 'retailer', 'balancing', 'fixed',
                                                             'price'], [0], self.__get_value_list],
                        'price_energy_balancing_negative': [0,
                                                            [self.market, 'pricing', 'retailer', 'balancing', 'fixed',
                                                             'price'], [1], self.__get_value_list],
                        'levy_pricing_mechanism': [0, [self.market, 'pricing', 'retailer', 'levies', 'method'],
                                                   self.__get_value],
                        'path_levy_prices': [0, [self.market, 'pricing', 'retailer', 'levies', 'file', 'file'],
                                             self.__get_value],
                        'price_energy_levies_positive': [0, [self.market, 'pricing', 'retailer', 'levies', 'fixed',
                                                             'price'], [0], self.__get_value_list],
                        'price_energy_levies_negative': [0, [self.market, 'pricing', 'retailer', 'levies', 'fixed',
                                                             'price'], [1], self.__get_value_list],
                    },
            },
            'retailer': {
                'sources': ['config/config_markets.yaml', 'general/retailer.ft'],
                'params': {
                        'retail_pricing_mechanism': [0, [self.market, 'pricing', 'retailer', 'energy', 'method'],
                                                     self.__get_value],
                        'path_retail_prices': [0, [self.market, 'pricing', 'retailer', 'energy', 'file', 'file'],
                                               self.__get_value],
                        'price_sell': [0, [self.market, 'pricing', 'retailer', 'energy', 'fixed', 'price'], [1],
                                       self.__get_value_list],
                        'price_buy': [0, [self.market, 'pricing', 'retailer', 'energy', 'fixed', 'price'], [0],
                                      self.__get_value_list],
                        'qty_energy_bid': [0, [self.market, 'pricing', 'retailer', 'energy', 'fixed', 'quantity'], [1],
                                           self.__get_value_list],
                        'qty_energy_offer': [0, [self.market, 'pricing', 'retailer', 'energy', 'fixed', 'quantity'],
                                             [0], self.__get_value_list],
                        'quality': 'na',
                        'id_user': 'retailer01',
                    },
            },
            'weather': {
                'sources': ['general/weather/weather.ft'],
            },
        }

    def convert(self):

        # Create output folder
        self._create_folder(path=self.output_path, delete=True)

        # Convert config
        self._convert_config()

        # Convert agents
        self._convert_agents()

        # Convert market
        self._convert_market()

        # Convert retailer
        self._convert_retailer()

        # Convert weather
        self._convert_weather()

    def _convert_config(self, name: str = 'config'):

        # Get relevant info
        info = self.naming[name]
        sources = info['sources']
        params = info['params']

        # Load sources
        files = []
        for source in sources:
            files.append(self._load_file(os.path.join(self.input_path, source)))

        # Add info to config
        config = {}
        self.__loop_through_dict(params, '', self._config_action, params=params, files=files, config=config)

        # Save config
        self._save_file(os.path.join(self.output_path, name + '.yaml'), config)

        return None

    def _convert_agents(self, name: str = 'agents'):

        # Create prosumer folder
        self._create_folder(path=os.path.join(self.output_path, 'prosumer'), delete=False)

        # Get relevant info
        info = self.naming[name]

        # Loop through all agent types
        for agent_type in info['type']:
            path = os.path.join(self.input_path, name, agent_type)
            agents = [x[0] for x in os.walk(path)][1:]
            # Loop through each agent in the agent type
            for idx, agent in enumerate(agents):
                if agent_type == 'sfh':
                    self._convert_sfh(agent, idx, info['sfh'])
                else:
                    raise NotImplementedError

    def _convert_market(self, name: str = 'market'):

        # Create prosumer folder
        self._create_folder(path=os.path.join(self.output_path, 'lem'), delete=False)

        # Get relevant info
        info = self.naming[name]
        sources = info['sources']
        params = info['params']

        # Load sources
        files = []
        for source in sources:
            files.append(self._load_file(os.path.join(self.input_path, source)))

        # Add info to config
        config = {}
        self.__loop_through_dict(params, '', self._config_action, params=params, files=files, config=config)

        # Save config
        self._save_file(os.path.join(self.output_path, 'lem', 'config_account.json'), config)

    def _convert_retailer(self, name: str = 'retailer'):

        # Create prosumer folder
        self._create_folder(path=os.path.join(self.output_path, 'retailer'), delete=False)

        # Get relevant info
        info = self.naming[name]
        sources = info['sources']
        params = info['params']

        # Load sources
        files = []
        for source in sources:
            files.append(self._load_file(os.path.join(self.input_path, source)))

        # Add info to config
        config = {}
        self.__loop_through_dict(params, '', self._config_action, params=params, files=files, config=config)

        # Save config
        self._save_file(os.path.join(self.output_path, 'retailer', 'config_account.json'), config)

        # Add retail price file
        df = files[1]
        df = df[['energy_price_buy', 'energy_price_sell']]
        df.columns = ['price_sell', 'price_buy']  # flip buy and sell as convention is different in lemlab
        self._save_file(os.path.join(self.output_path, 'retailer', 'retail_prices.ft'), df)

    def _convert_weather(self, name: str = 'weather'):

        # Create prosumer folder
        self._create_folder(path=os.path.join(self.output_path, 'weather'), delete=False)

        # Get relevant info
        info = self.naming[name]
        source = info['sources'][0]

        # Load source
        df = self._load_file(os.path.join(self.input_path, source))

        # Save config
        self._save_file(os.path.join(self.output_path, 'weather', 'weather.ft'), df)

    def _convert_sfh(self, agent: str, idx: int, info: dict):

        # Create agent folder
        id = str(idx + 1).zfill(10)
        self._create_folder(path=os.path.join(self.output_path, 'prosumer', id), delete=False)

        # Load files
        id_old = agent.rsplit('\\', 1)[-1]
        files = {
            'account.json': {
                'dest': 'config_account.json',
                'file': self._load_file(os.path.join(agent, 'account.json'))
            },
            'plants.json': {
                'dest': 'config_plants.json',
                'file': self._load_file(os.path.join(agent, 'plants.json'))
            },
            'meters.ft': {
                'file': self._load_file(os.path.join(agent, 'meters.ft'))
            },
            'socs.ft': {
                'file': self._load_file(os.path.join(agent, 'socs.ft'))
            },
            'timeseries.ft': {
                'file': self._load_file(os.path.join(agent, 'timeseries.ft'))
            },
        }

        # Loop through all files
        for file, category in info.items():
            # Create output dictionary (not for every file required)
            output = {}
            if file == 'account.json':
                # Loop through categories
                for key, val in category.items():
                    for lemlab, hamlet in val.items():
                        if hamlet['src']:
                            try:
                                output[lemlab] = files[file]['file'][key][hamlet['src']]
                            except TypeError:
                                output[lemlab] = files[file]['file'][key]
                            if hamlet['func']:
                                output[lemlab] = hamlet['func'](output[lemlab], id=id)
                            if hamlet['type']:
                                output[lemlab] = hamlet['type'](output[lemlab])
                        else:
                            if callable(hamlet['func']):
                                try:
                                    output[lemlab] = hamlet['func'](output[lemlab], id=id)
                                except KeyError:
                                    output[lemlab] = hamlet['func'](id=id, path=os.path.join(self.output_path, 'prosumer', id))
                            else:
                                output[lemlab] = hamlet['func']
                            if hamlet['type']:
                                output[lemlab] = hamlet['type'](output[lemlab])

                # Save output
                self._save_file(os.path.join(self.output_path, 'prosumer', id, files[file]['dest']), output)

            elif file == 'plants.json':
                # Loop through categories
                for key, val in category.items():
                    # Get source plant from file
                    plant = next((item for id, item in files[file]['file'].items() if item['type'] == val['src']), None)
                    plant_id = next((id for id, item in files[file]['file'].items() if item['type'] == val['src']), None)
                    if plant is None:
                        continue
                    output[plant_id] = {}
                    for lemlab, hamlet in val['params'].items():
                        if hamlet['src']:
                            try:
                                output[plant_id][lemlab] = plant[hamlet['src']]
                            except TypeError:
                                out_val = plant
                                for _, idx_item in enumerate(hamlet['src']):
                                    out_val = out_val[idx_item]
                                output[plant_id][lemlab] = out_val
                            if hamlet['func']:
                                output[plant_id][lemlab] = hamlet['func'](output[plant_id][lemlab], id=id)
                            if hamlet['type']:
                                output[plant_id][lemlab] = hamlet['type'](output[plant_id][lemlab])
                        else:
                            if callable(hamlet['func']):
                                output[plant_id][lemlab] = hamlet['func'](output[plant_id][lemlab], id=id)
                            else:
                                output[plant_id][lemlab] = hamlet['func']
                            if hamlet['type']:
                                output[plant_id][lemlab] = hamlet['type'](output[plant_id][lemlab])

                # Save output
                self._save_file(os.path.join(self.output_path, 'prosumer', id, files[file]['dest']), output)

            elif file == 'meters.ft':
                # Split meter file into separate meter files
                category(file=files[file]['file'], dest=os.path.join(self.output_path, 'prosumer', id))

            elif file == 'socs.ft':
                # Split soc file into separate soc files
                category(file=files[file]['file'], dest=os.path.join(self.output_path, 'prosumer', id))

            elif file == 'timeseries.ft':
                # Split soc file into separate timeseries files
                category(file=files[file]['file'], dest=os.path.join(self.output_path, 'prosumer', id))

            else:
                raise NotImplementedError

    @staticmethod
    def _conv_idx(val, **kwargs):
        """Converts an index to a string"""

        val = kwargs['id']

        return val

    def _create_main_meter(self, **kwargs):
        """Creates a main meter file and returns the meter id"""

        path = kwargs['path']

        # Create meter id
        meter_id = self.__gen_rand_id(10)

        # Create meter file
        with open(f"{path}/meter_{meter_id}.json", "w+") as write_file:
            json.dump([0, 0], write_file)

        return meter_id


    @staticmethod
    def _divide_by_1000(val, **kwargs):
        """Divides a value by 1000"""

        return val / 1000

    @staticmethod
    def _conv_str_to_list(val, **kwargs) -> list:
        """Converts a string to a list

        Args:
            val: string to convert
            kwargs: not used

        Returns:
            list

        """

        return ast.literal_eval(val)

    @staticmethod
    def _split_meter(file: pd.DataFrame, dest: str):
        """Splits the meter file into separate meter files

        Args:
            file: meter file
            dest: destination folder

        Returns:
            None

        """

        # Loop through all meters by column
        for meter in file.columns:
            readings = [file[meter].iloc[0], file[meter].iloc[1]]

            # Save meter readings as json file
            with open(os.path.join(dest, f'meter_{meter}.json'), 'w') as f:
                json.dump(readings, f, cls=NpEncoder)

    @staticmethod
    def _split_soc(file: pd.DataFrame, dest: str):
        """Splits the soc file into separate soc files

        Args:
            file: soc file
            dest: destination folder

        Returns:
            None

        """

        # Loop through all socs by column
        for soc in file.columns:
            readings = int(file[soc].iloc[0])

            # Save soc readings as json file
            with open(os.path.join(dest, f'soc_{soc}.json'), 'w') as f:
                json.dump(readings, f, cls=NpEncoder)

    @staticmethod
    def _split_timeseries(file: pd.DataFrame, dest: str):
        """Splits the timeseries file into separate timeseries files

        Args:
            file: timeseries file
            dest: destination folder

        Returns:
            None

        """

        # Get unique IDs from the column names
        unique_ids = set([col.split("_")[0] for col in file.columns])

        # Loop through the unique IDs
        for uid in unique_ids:
            # Get columns that belong to the current ID
            cols = [col for col in file.columns if col.startswith(uid)]

            # Create a new dataframe with only those columns
            new_df = file[cols]

            # Rename columns to remove the ID prefix
            new_df.columns = [col.split("_", 1)[1] for col in cols]

            # Save the new dataframe to a file
            new_df = new_df.reset_index()
            new_df.to_feather(os.path.join(dest, f"raw_data_{uid}.ft"))

    def __loop_through_dict(self, nested_dict: dict, path: str, func: callable, *args, **kwargs) -> None:
        """loops through the dictionary and calls the function for each item

        Args:
            nested_dict: dictionary to loop through
            path: path to the folder
            func: function to call
            *args: arguments for the function
            **kwargs: keyword arguments for the function

        Returns:
            None

        """

        # Loop through all key-value pairs of the dictionary
        for key, value in nested_dict.items():
            try:
                # Check if value is a dictionary
                if isinstance(value, dict):
                    # If value is a dictionary, go one level deeper
                    output = self.__loop_through_dict(value, ','.join([path, key]), func, *args, **kwargs)
                else:
                    # If value is not a dictionary, call the function
                    output = func(','.join([path, key]), *args, **kwargs)
            except TypeError:
                # If there's a TypeError, go up a level and call the function
                output = func(path, *args, **kwargs)

        return output

    @staticmethod
    def _config_action(path: str, params: dict, files: list, config: dict, *args, **kwargs) -> dict:

        # Get the keys in the path
        keys = path.split(',', 1)[1].split(',')

        # Get the value of the dictionary
        for key in keys:
            params = params[key]
        # print(params, isinstance(params, list))

        # Add the value to the nested config
        current_dict = config
        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]

        lowest_key = keys[-1]
        if isinstance(params, list):
            # If the value is a list, get the value from the file
            file = files[params[0]]
            if callable(params[2]):
                pos = None
            else:
                pos = params[2][0]
            current_dict[lowest_key] = params[-1](file=file, keys=params[1], pos=pos)
        else:
            # If the value is not a list, add the value
            current_dict[lowest_key] = params

        return config

    @staticmethod
    def __config_sim_start(file, keys, *args, **kwargs) -> str:
        # Get the value of the dictionary
        for key in keys:
            file = file[key]

        return str(file + pd.Timedelta(hours=1))

    @staticmethod
    def __get_value(file, keys, *args, **kwargs):
        # Get the value of the dictionary
        for key in keys:
            file = file[key]

        return file

    @staticmethod
    def __get_value_list(file, keys, pos, *args, **kwargs):
        # Get the value of the dictionary
        for key in keys:
            file = file[key]

        return file[pos]

    @staticmethod
    def _load_file(path: str, index: int = 0):
        file_type = path.rsplit('.', 1)[-1]
        if file_type == 'yaml' or file_type == 'yml':
            with open(path) as file:
                file = YAML().load(file)
        elif file_type == 'json':
            with open(path) as file:
                file = json.load(file)
        elif file_type == 'csv':
            file = pd.read_csv(path, index_col=index)
        elif file_type == 'xlsx':
            file = pd.ExcelFile(path)
        elif file_type == 'ft':
            file = pd.read_feather(path)
            if index is not None:
                file.set_index(file.columns[index], inplace=True, drop=True)
        else:
            raise ValueError(f'File type "{file_type}" not supported')

        return file

    @staticmethod
    def _save_file(path: str, data, index: bool = True) -> None:
        file_type = path.rsplit('.', 1)[-1]

        if file_type == 'yaml' or file_type == 'yml':
            with open(path, 'w') as file:
                YAML().dump(data, file)
        elif file_type == 'json':
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)
        elif file_type == 'csv':
            data.to_csv(path, index=index)
        elif file_type == 'xlsx':
            data.to_excel(path, index=index)
        elif file_type == 'ft':
            data.reset_index(inplace=True)
            data.to_feather(path)
        else:
            raise ValueError(f'File type "{file_type}" not supported')

    @staticmethod
    def _create_folder(path: str, delete: bool = True) -> None:
        """Creates a folder at the given path

        Args:
            path: path to the folder
            delete: if True, the folder will be deleted if it already exists

        Returns:
            None
        """

        # Create main folder if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if delete:
                shutil.rmtree(path)
                os.makedirs(path)
        time.sleep(0.0001)
    @staticmethod
    def __gen_rand_id(length: int) -> str:
        """generates a random combination of ascii characters and digits of specified length

        Args:
            length: integer specifying the length of the string

        Returns:
            string with length equal to input argument length

        """

        characters = string.ascii_lowercase + string.digits * 3

        return ''.join(choice(characters) for _ in range(length))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    h2l = h2l()
    h2l.convert()
