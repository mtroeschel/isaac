import os.path

import mosaik


SIM_CONFIG = {
    'ChpSim': {
        'python': 'chpsim.mosaik:ChpSim',
    },
    'MAS': {
        'cmd': 'openvpp-mosaik -l info %(addr)s',
    },
    'DB': {
        'cmd': 'mosaik-hdf5 %(addr)s',
    },
}

START_DATE = '2010-04-08T00:00:00+02:00'
# END = 2.5 * 24 * 3600  # 2 days
END = 1 * 24 * 3600  # 1 day

N_CHPS = 3
CHP_TYPE = '50_kw_el'

START_AGENTS = True

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOG_DIR = os.path.dirname(__file__)

CHP_TYPES = {
    'default': {
        'chp_p_levels': [(0, 0), (1000, 2000)],
        'chp_min_off_time': 15,
        'storage_e_th_init': .5,
        'storage_e_th_max': 0,
    },
    '5_5_kw_el': {
        'chp_p_levels': [(0, 0), (5500, 12700)],
        'chp_min_off_time': 15,
        'storage_e_th_init': 12700,
        'storage_e_th_max': 25400,
    },
    '4_7_kw_el': {
        'chp_p_levels': [(0, 0), (4700, 12500)],
        'chp_min_off_time': 15,
        'storage_e_th_init': 12500,
        'storage_e_th_max': 25000,
    },
    '50_kw_el': {
        'chp_p_levels': [(0, 0), (49950, 99900)],
        'chp_min_off_time': 30,
        'storage_e_th_init': 47500,
        'storage_e_th_max': 95000,
    },
}
CHP_DEMAND_FILE = os.path.join(DATA_DIR, 'demand.csv.xz')
TARGET_DIR = os.path.join(DATA_DIR, 'targets')
RESULT_DIR = os.path.join(LOG_DIR, 'results')

MINUTE = 60
AGENT_CONFIG = {
    'MosaikAgent': {
    },
    'UserAgent': {
        'tz': 'Europe/Berlin',
        'demand_file': CHP_DEMAND_FILE,
        'target_dir': TARGET_DIR,
        'result_dir': RESULT_DIR,
        'dap_planning_horizon': {'days': 1},
        'dap_res': 15 * MINUTE,
        'dap_start': {'hour': 9},
    },
    'ControllerAgent': {
        'n_agents': N_CHPS,
        'topology_phi': 1,
        'topology_seed': 23,
        'negotiation_single_start': True,
        'negotiation_timeout': 30 * MINUTE,
    },
    'ObserverAgent': {
        'n_agents': N_CHPS,
        'negotiation_timeout': 30 * MINUTE,
        'log_dbfile': os.path.join(LOG_DIR, 'openvpp-agents.hdf5'),
    },
    'Planner': {
        'check_inbox_interval': .1,  # [s]
    },
}


def main():
    world = mosaik.World(SIM_CONFIG)
    create_scenario(world)
    world.run(until=END)


def create_scenario(world):
    chps = setup_chps(world)

    if START_AGENTS:
        setup_mas(world, chps)

    setup_database(world, chps)


def setup_chps(world):
    """Set-up the ChpSim and return a list of the CHP entities."""
    chpsim = world.start('ChpSim', start_date=START_DATE,
                         demand_file=CHP_DEMAND_FILE)
    chps = chpsim.CHP.create(N_CHPS, **CHP_TYPES[CHP_TYPE])
    return chps


def setup_mas(world, chps):
    """Set-up the multi-agent system"""
    mas = world.start('MAS', start_date=START_DATE, n_agents=len(chps),
                      config=AGENT_CONFIG)

    chp_data = world.get_data(chps, 'chp_p_levels', 'chp_min_off_time',
                              'storage_e_th_max')
    model_conf = {
        e.full_id: data for e, data in chp_data.items()
    }
    agents = mas.Agent.create(len(chp_data), model_conf=model_conf)

    for chp, agent in zip(chps, agents):
        world.connect(chp, agent, 'chp_p_el', 'storage_e_th',
                      async_requests=True)


def setup_database(world, chps):
    db = world.start('DB', step_size=60, duration=END)
    suffix = 'controlled' if START_AGENTS else 'uncontrolled'
    dbfile = os.path.join(LOG_DIR, 'results_%s.hdf5' % suffix)
    hdf5 = db.Database(filename=dbfile)
    mosaik.util.connect_many_to_one(world, chps, hdf5,
                                    'chp_p_el', 'storage_e_th')


if __name__ == '__main__':
    main()
