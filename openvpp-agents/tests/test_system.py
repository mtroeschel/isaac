"""
Simulate two days (2nd and 3rd January 2015) based on forecast data
from 1st to 5th January.

We skip the first day (just to see if it works without problems).  We also
need forecast data for two additonal days (thus until the 5th January) because
we have a look-ahead period of 48h.

"""
from asyncio import coroutine
import asyncio
import json
import logging

import aiomas
import arrow
import chpsim.mosaik as chpsim
import h5py
import numpy as np
import pytest


pytestmark = pytest.mark.system


CHP_CONF = {
    'chp_p_levels': [(0, 0), (5000, 12000)],
    'chp_min_off_time': 30,
    'storage_e_th_max': 24000,
}
START_DATE = '2015-01-02T00:00:00+01:00'
DURATION = 2  # Simulate two days
N_CHPS = 3
ADDR = ('127.0.0.1', 5555)
INIT_DEMAND = 2000  # Thermal demand for the first day
DEMAND_INC = 500  # Increased by this amount each day

REFERENCE_DBFILE = __file__.replace('.py', '.hdf5')
EXPECTED_SETPOINTS = dict([
    (3660, {  # day 1, 01:01
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 0.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 0.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 0.0}}}),
    (15600, {  # day 1, 04:20
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 5000.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 5000.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 5000.0}}}),
    (24300, {  # day 1, 06:45
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 0.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 0.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 0.0}}}),
    (57360, {  # day 1, 15:56
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 5000.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 5000.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 5000.0}}}),
    (66060, {  # day 1, 18:21
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 0.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 0.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 0.0}}}),
    (96960, {  # day 2, 02:56
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 5000.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 5000.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 5000.0}}}),
    (106140, {  # day 2, 05:29
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 0.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 0.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 0.0}}}),
    (133680, {  # day 2, 13:08
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 5000.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 5000.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 5000.0}}}),
    (142860, {  # day 2, 15:41
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 0.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 0.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 0.0}}}),
    (170400, {  # day 2, 23:20
        'chp-0': {'mas-0.Agent_0': {'chp_p_el': 5000.0}},
        'chp-1': {'mas-0.Agent_1': {'chp_p_el': 5000.0}},
        'chp-2': {'mas-0.Agent_2': {'chp_p_el': 5000.0}}}),
])


@pytest.fixture
def demand_file(tmpdir):
    start_time = arrow.get(START_DATE).to('utc').replace(days=-1)
    demand_meta = {
        'start_time': str(start_time),
        'interval_minutes': 1,
        'cols': ['p_el', 'p_th_heat', 'p_th_water'],
    }
    demand_data = []
    for i in range(DURATION + 3):
        demand_data += [(0, INIT_DEMAND + i * DEMAND_INC, 0)] * 1440
    lines = [','.join(map(str, row)) for row in demand_data]
    lines = '\n'.join(lines)
    content = '{}\n{}\n'.format(json.dumps(demand_meta), lines)
    demand_file = tmpdir.join('test_data.csv')
    demand_file.write(content)
    return demand_file.strpath


@pytest.fixture
def target_dir(tmpdir):
    target_meta = {
        'start_time': '2009-12-31T23:00:00+00:00',
        'interval_minutes': 60,
        'cols': ['p_el', 'weight'],
    }
    target_data = [(0, 1)] * 1440 * (DURATION + 3)
    lines = [','.join(map(str, row)) for row in target_data]
    lines = '\n'.join(lines)
    content = '{}\n{}\n'.format(json.dumps(target_meta), lines)

    tfd = tmpdir.mkdir('targets')
    tf = tfd.join('electricaltarget1.csv')
    tf.write(content)
    return tfd.strpath


@pytest.fixture
def agent_conf(demand_file, target_dir, tmpdir):
    return {
        'MosaikAgent': {
        },
        'UserAgent': {
            'tz': 'Europe/Berlin',
            'demand_file': demand_file,
            'target_dir': target_dir,
            'dap_planning_horizon': {'days': 1},
            'dap_res': 15 * 60,
            'dap_start': {'hour': 1},
        },
        'ControllerAgent': {
            'n_agents': N_CHPS,
            'topology_phi': 1,
            'topology_seed': 23,
            'negotiation_single_start': True,
            'negotiation_timeout': 30 * 60,
        },
        'ObserverAgent': {
            'n_agents': N_CHPS,
            'negotiation_timeout': 30 * 60,
            'log_dbfile': tmpdir.join('openvpp-agents.hdf5').strpath,
        },
        'Planner': {
            'check_inbox_interval': .01,  # [s]
        },
    }


class MosaikMock:
    router = aiomas.rpc.Service()
    setpoints = {}

    @router.expose
    def get_related_entities(self, aids):
        assert aids == ['mas-0.Agent_{}'.format(i) for i in range(N_CHPS)]
        return {aid: {'chp-{}'.format(i): 'CHP'}
                for i, aid in enumerate(aids)}

    @router.expose
    def set_data(self, data):
        setpoints = {}
        for aid, unit_data in sorted(data.items()):
            assert len(unit_data) == 1
            uid, values = unit_data.popitem()
            p_el = values['chp_p_el']
            # p_el_levels = np.array([l[0] for l in CHP_CONF.chp_p_levels])
            # sp = np.argmin(np.abs(p_el_levels - p_el))
            # setpoints.append(sp)
            setpoints[uid] = {aid: {'chp_p_el': p_el}}

        self.setpoints = setpoints


@coroutine
def run_test(mosaik_con, agent_conf, mosaik_mock, demand_file):
    mas = mosaik_con.remote
    try:
        yield from _init(mas, agent_conf)
        yield from _create(mas)
        yield from _setup_done(mas)
        yield from _step(mas, mosaik_mock, demand_file)
    finally:
        yield from _stop(mas)


@pytest.mark.asyncio
def test_simulation(demand_file, agent_conf, tmpdir, event_loop):
    logging.basicConfig(level=logging.DEBUG)
    mosaik_mock = MosaikMock()

    # Setup and start the fake mosaik environment that performs the test:
    def cb(con):
        print('Starting test ...')
        mosaik_mock.test_task = aiomas.async(
            run_test(con, agent_conf, mosaik_mock, demand_file))

    server_sock = yield from aiomas.rpc.start_server(
        ADDR, mosaik_mock, cb)

    try:
        # Start the MAS :
        mas_proc = yield from asyncio.create_subprocess_shell(
            'openvpp-mosaik -l debug %s:%d' % ADDR)

        print('Waiting for MAS...')
        yield from mas_proc.wait()  # Wait for the MAS
        print('Waiting for fake mosaik...')
        yield from mosaik_mock.test_task  # Wait for the fake mosaik
    finally:
        server_sock.close()
        yield from server_sock.wait_closed()

    results_db = h5py.File(agent_conf['ObserverAgent']['log_dbfile'], 'r')
    expected_db = h5py.File(REFERENCE_DBFILE, 'r')
    special = {
       'dap_data': _compare_dap_data,
       'perf_data': _compare_perf_data,
    }
    _compare_hdf5(results_db, expected_db, special)


@coroutine
def _init(mas, agent_conf):
    print('init')
    meta = yield from mas.init('mas-0', start_date=START_DATE,
                               n_agents=N_CHPS, config=agent_conf)
    assert meta == {
        'api_version': '2.2',
        'models': {'Agent': {
            'attrs': ['chp_p_el', 'storage_e_th'],
            'params': ['model_conf'],
            'public': True}}}


@coroutine
def _create(mas):
    print('create')
    model_conf = {'chp-{}'.format(i): CHP_CONF for i in range(N_CHPS)}
    entities = yield from mas.create(N_CHPS, 'Agent', model_conf=model_conf)
    assert list(sorted(entities, key=lambda i: i['eid'])) == [
        {'type': 'Agent', 'eid': 'Agent_0'},
        {'type': 'Agent', 'eid': 'Agent_1'},
        {'type': 'Agent', 'eid': 'Agent_2'},
    ]


@coroutine
def _setup_done(mas):
    print('setup_done')
    ret = yield from mas.setup_done()
    assert ret is None


@coroutine
def _step(mas, mosaik, demand_file):
    print('step')

    # Setup CHPs
    conf = dict(CHP_CONF)
    conf['storage_e_th_init'] = 0.5 * conf['storage_e_th_max']
    chp_sim = chpsim.ChpSim()
    chp_sim.init('chpsim', start_date=START_DATE, demand_file=demand_file)
    entities = chp_sim.create(N_CHPS, 'CHP', **conf)
    chp_sim.setup_done()

    # Step through the sim
    for t in range(0, DURATION * 24 * 3600, 60):
        # print('------------- step %d' % t)
        expected = EXPECTED_SETPOINTS.get(t, {})
        assert mosaik.setpoints == expected

        chp_sim.step(t, mosaik.setpoints)
        mosaik.setpoints = {}
        data = chp_sim.get_data({e['eid']: ['chp_p_el', 'storage_e_th']
                                 for e in entities})

        inputs = {
            'Agent_%d' % i: {
                attr: {eid: value} for attr, value in attrs.items()
            } for i, (eid, attrs) in enumerate(sorted(data.items()))
        }
        ret = yield from mas.step(t, inputs)
        assert ret == t + 60


@coroutine
def _stop(mas):
    print('stop')
    with pytest.raises(ConnectionResetError):
        yield from mas.stop()


def _compare_hdf5(a, b, special):
    assert type(a) == type(b), \
        'Type mismatch at %s: %s != %s' % (a, type(a), type(b))

    a_attrs = dict(a.attrs.items())
    b_attrs = dict(b.attrs.items())
    assert a_attrs == b_attrs, \
        'Attribute mismatch at %s: %s != %s' % (a, a_attrs, b_attrs)

    name = a.name.rsplit('/', 1)[-1]
    if name in special:
        special[name](a, b)
    elif isinstance(a, h5py.Dataset):
        assert len(a) == len(b), \
            'Entry count mismatch at %s: %s != %s' % (a, len(a), len(b))
        #
        # if len(a) == 0:
        #     # Empty datasets cannot be accessed.
        #     return

        assert np.array_equal(a[:], b[:]), \
            'Dataset mismatch at %s: %s != %s' % (a, a[:], b[:])
    else:
        assert set(a) == set(b), 'Group mismatch at %s: %s' % (
            a, ', '.join(sorted(set(a) ^ set(b))))

        for name in sorted(a):
            _compare_hdf5(a[name], b[name], special)


def _compare_dap_data(a, b):
    a_start = a[0][0]
    a_end = a[-1][0]
    a_time = a_end - a_start
    b_start = b[0][0]
    b_end = b[-1][0]
    b_time = b_end - b_start
    t_diff = a_time - b_time
    if t_diff < 0:
        print('Test run was %.3fs (%.2f%%) faster then reference (%.3fs).' %
              (-t_diff, -t_diff * 100 / b_time, b_time))
    if t_diff > 0:
        print('Test run was %.3fs(%.2f%%) slower then reference (%.3fs).' %
              (t_diff, t_diff * 100 / b_time, b_time))
    # The test is so short, that we cannot really compare performance
    # assert abs(t_diff) < 0.5 * b_time


def _compare_perf_data(a, b):
    a_max_mem = max(int(mem) for t, cpu, mem in a[:]) if len(a[:]) else 0
    b_max_mem = max(int(mem) for t, cpu, mem in b[:]) if len(b[:]) else 0
    mem_diff = a_max_mem - b_max_mem
    if mem_diff < 0:
        print('Test run used %.2fMiB (%.2f%%) less then reference (%.2fMiB).' %
              (-mem_diff / (1024 ** 2),
               -mem_diff * 100 / b_max_mem,
               b_max_mem / (1024 ** 2)))
    if mem_diff > 0:
        print('Test run used %.2fMiB (%.2f%%) more then reference (%.2fMiB).' %
              (mem_diff / (1024 ** 2),
               mem_diff * 100 / b_max_mem,
               b_max_mem / (1024 ** 2)))
    # The test is so short, that we cannot really compare memory consumption
    # assert abs(mem_diff) < 0.1 * b_max_mem
