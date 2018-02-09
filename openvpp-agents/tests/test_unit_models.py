from collections import deque

import arrow
import numpy as np
import pytest

from openvpp_agents import unit_models
from openvpp_agents.util import TimeSeries as TS


@pytest.fixture
def model():
    return unit_models.ChpModel([(0, 0), (2, 60), (6, 120)], 10, 8)


@pytest.mark.parametrize(['p_el', 'expected'], [
    (-1, 0),
    (0, 0),
    (2, 1),
    (2.1, 1),
    (6, 2),
    (8, 2),
])
def test_chp_model_setpoint_for(model, p_el, expected):
    assert model._setpoint_for(p_el) == expected


def test_chp_model_get_schedule(model):
    start = arrow.get('2015-01-01')
    sched = TS(start, 60, [0, 1, 2, 3])
    model._schedules = {0: sched}
    assert model.get_schedule(0) == sched


@pytest.mark.parametrize('res', [30, 60, 900])
def test_chp_model_update_forecast(model, res):
    fc = TS(arrow.get(), res, [0, 1, 2, 3])
    model.update_forecast(fc)
    assert model._forecast_demand_p_th == fc


@pytest.mark.parametrize('res', [23, 70])
def test_chp_model_update_forecast_wrong_resolution(model, res):
    """An error should be raised if the updated forecast has not the expected
    resolution."""
    assert model.model.res == 60
    fc = TS(arrow.get(), res, [0, 1, 2, 3])
    pytest.raises(ValueError, model.update_forecast, fc)


def test_chp_model_generate_schedules(model):
    start = arrow.get('2015-01-01')
    res = 60
    demand_p_th = np.array([0, 30, 60, 60, 30, 0, 60, 60, 0, 0])
    fc = TS(start, res, demand_p_th)
    model.update_forecast(fc)

    # minimal state must contain 2 entries
    state = deque(maxlen=2)
    state.append((start.replace(seconds=-60),
                  {'chp_p_el': 0, 'storage_e_th': .5}))
    state.append((start, {'chp_p_el': 0, 'storage_e_th': .5}))
    ret = model.generate_schedules(start, res=2 * 60, intervals=5, state=state)

    print(ret)
    assert len(ret) == 1
    assert ret[0][:2] == (0, 1)
    assert np.array_equal(ret[0][2], [0, 6, 6, 6, 3])

    assert model._schedules == {
        0: TS(start, 60, [0, 0, 6, 6, 6, 6, 6, 6, 6, 0]),
    }


def test_chp_model_generate_schedules_wrong_resolution(model):
    """An error should be raised if the schedule resolution is lower then the
    model resolution or if it is not a multiple the model resolution."""
    assert model.model.res == 60
    pytest.raises(ValueError, model.generate_schedules, None, 30, None, None)
    pytest.raises(ValueError, model.generate_schedules, None, 62, None, None)


def test_chp_reset():
    """Test resetting the CHP model."""
    chp = unit_models.CHP([(0, 0), (1, 2), (2, 4)], 10, 1, 0)
    assert chp.chp_setpoint == 0
    assert chp.chp_off_since == 10
    assert chp.storage_e_th == 0

    chp.reset(chp_setpoint=2)
    assert chp.chp_setpoint == 2
    assert chp.chp_off_since == 10
    assert chp.storage_e_th == 0

    chp.reset(chp_setpoint=0, storage_e_th=1)
    assert chp.chp_setpoint == 0
    assert chp.chp_off_since == 0  # Reset to 0
    assert chp.storage_e_th == 1

    chp.chp_off_since = 3
    chp.reset(chp_setpoint=1, storage_e_th=0.5)
    assert chp.chp_setpoint == 1
    assert chp.chp_off_since == 3  # No reset!
    assert chp.storage_e_th == .5

    chp.reset(storage_e_th=1)
    assert chp.chp_setpoint == 1
    assert chp.chp_off_since == 3  # No reset!
    assert chp.storage_e_th == 1


@pytest.mark.parametrize(['setpoint', 'off_since', 'new_sp', 'exp_sp',
                          'exp_off_since', 'exp_p_el', 'exp_p_th'], [
    (1, 10, None, 1, 10, 1, 2),  # Don't actually set a setpoint -> no change
    (0,  9,    2, 0,  9, 0, 0),  # Turn on CHP but it was not off long enough -> no change  # NOQA
    (0, 10,    2, 2, 10, 2, 4),  # Turn on CHP which was off long enough -> turned on  # NOQA
    (0, 10,    0, 0, 10, 0, 0),  # Turn off non-running CHP -> turned off
    (1, 10,    2, 2, 10, 2, 4),  # Raise setpoint for running CHP -> setpoint adjusted  # NOQA
    (1, 10,    0, 0,  0, 0, 0),  # Turn off running CHP -> turned off
])
def test_chp_set_setpoint(setpoint, off_since, new_sp, exp_sp, exp_off_since,
                          exp_p_el, exp_p_th):
    """Test setting setpoints ot the CHP."""
    chp = unit_models.CHP([(0, 0), (1, 2), (2, 4)], 10, 1, 0)
    chp.chp_setpoint = setpoint
    chp.chp_off_since = off_since

    chp.set_setpoint(new_sp)

    assert chp.chp_setpoint == exp_sp
    assert chp.chp_off_since == exp_off_since
    assert chp.chp_p_el == exp_p_el
    assert chp.chp_p_th == exp_p_th


@pytest.mark.parametrize(['storage_e', 'setpoint', 'exp_sp'], [
    (1, 1, 0),  # Storage full, on -> off
    (1, 0, 0),  # Storage full, off -> off
    (0, 1, 1),  # Storage empty, on -> on (no change to higher setpoint!)
    (0, 0, 2),  # Storage empty, off -> on (full power)
    (.5, 1, 1),  # Storage medium, on -> no change
    (.5, 0, 0),  # Storage medium, off -> no change
])
def test_chp_status(storage_e, setpoint, exp_sp):
    """Test if the controllers turns on/off the CHP according to the current
    storage energy."""
    chp = unit_models.CHP([(0, 0), (1, 2), (2, 4)], 10, 1, 0)
    chp.chp_setpoint = setpoint
    chp.storage_e_th = storage_e

    chp.check_chp_status()
    assert chp.chp_setpoint == exp_sp


def test_step_basic():
    """Test normal simulation."""
    chp = unit_models.CHP([(0, 0), (1, 60), (2, 120)], 10, 30, 0)
    chp.reset(storage_e_th=2)
    setpoints = {
        10: 1,
        12: 2,
        35: 0,
        42: 1,
        44: 2,
        47: 0,
    }

    data = []
    for i in range(50):
        setpoint = setpoints.get(i, None)  # Check if I got a setpoint
        chp.step(60, setpoint)
        data.append((
            i,
            chp.chp_setpoint,
            chp.chp_off_since,
            chp.storage_e_th,
        ))

    assert data == [
        (0,  0, 11, 1.0),
        (1,  0, 12, 0.0),  # Storage empty, engine turned on
        (2,  2, 13, 1.0),
        (3,  2, 14, 2.0),
        (4,  2, 15, 3.0),
        (5,  2, 16, 4.0),
        (6,  2, 17, 5.0),
        (7,  2, 18, 6.0),
        (8,  2, 19, 7.0),
        (9,  2, 20, 8.0),
        (10, 1, 21, 8.0),  # Reduce power output for a while
        (11, 1, 22, 8.0),
        (12, 2, 23, 9.0),  # Reset setpoint to max.
        (13, 2, 24, 10.0),
        (14, 2, 25, 11.0),
        (15, 2, 26, 12.0),
        (16, 2, 27, 13.0),
        (17, 2, 28, 14.0),
        (18, 2, 29, 15.0),
        (19, 2, 30, 16.0),
        (20, 2, 31, 17.0),
        (21, 2, 32, 18.0),
        (22, 2, 33, 19.0),
        (23, 2, 34, 20.0),
        (24, 2, 35, 21.0),
        (25, 2, 36, 22.0),
        (26, 2, 37, 23.0),
        (27, 2, 38, 24.0),
        (28, 2, 39, 25.0),
        (29, 2, 40, 26.0),
        (30, 2, 41, 27.0),
        (31, 2, 42, 28.0),
        (32, 2, 43, 29.0),
        (33, 2, 44, 30.0),  # Storage full, engine turned off
        (34, 0, 1,  29.0),
        (35, 0, 2,  28.0),
        (36, 0, 3,  27.0),
        (37, 0, 4,  26.0),
        (38, 0, 5,  25.0),
        (39, 0, 6,  24.0),
        (40, 0, 7,  23.0),
        (41, 0, 8,  22.0),
        (42, 0, 9,  21.0),
        (43, 0, 10, 20.0),  # Cannot turn engine on yet
        (44, 2, 11, 21.0),  # Now we can
        (45, 2, 12, 22.0),
        (46, 2, 13, 23.0),
        (47, 0, 1,  22.0),  # And we turn it off again
        (48, 0, 2,  21.0),
        (49, 0, 3,  20.0),
    ]
