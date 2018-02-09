import os.path

import pytest

from chpsim.mosaik import ChpSim


def test_chpsim():
    chpsim = ChpSim()
    chpsim.init('chpsim-0', '2010-01-01T00:02:00+01:00',
                os.path.join(os.path.dirname(__file__), 'test_data.csv'))

    assert not chpsim._chps
    assert not chpsim._chp_config

    ret = chpsim.create(2, 'CHP',
                        chp_p_levels=[(0, 0), (1, 60), (2, 120)],
                        chp_min_off_time=10,
                        storage_e_th_max=30,
                        storage_e_th_init=2)

    assert ret == [
        {'eid': 'chp-0', 'type': 'CHP'},
        {'eid': 'chp-1', 'type': 'CHP'},
    ]

    assert len(chpsim._chps) == 2
    assert len(chpsim._chp_config) == 2

    ret = chpsim.create(1, 'CHP',
                        chp_p_levels=[(0, 0), (2, 30), (4, 60)],
                        chp_min_off_time=5,
                        storage_e_th_max=15,
                        storage_e_th_init=1)

    assert ret == [
        {'eid': 'chp-2', 'type': 'CHP'},
    ]

    assert len(chpsim._chps) == 3
    assert len(chpsim._chp_config) == 3

    chpsim.setup_done()

    ret = chpsim.step(0, {
        'chp-0': {},
        'chp-1': {'chp_p_el': {'src': 3.9}},
        'chp-2': {'chp_p_el': {'src': 2}},
    })

    assert ret == 60

    ret = chpsim.get_data({
        'chp-0': ['chp_setpoint', 'storage_e_th'],
        'chp-1': ['chp_setpoint', 'storage_e_th'],
        'chp-2': ['chp_setpoint', 'storage_e_th'],
    })

    assert ret == {
        'chp-0': {'chp_setpoint': 0, 'storage_e_th': 1},
        'chp-1': {'chp_setpoint': 2, 'storage_e_th': 3},
        'chp-2': {'chp_setpoint': 1, 'storage_e_th': .5},
    }

    pytest.raises(StopIteration, chpsim.step, 1, {})
