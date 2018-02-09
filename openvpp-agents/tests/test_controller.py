import asyncio
import collections
from asyncio import coroutine, gather

import aiomas

import arrow

import numpy as np

from openvpp_agents import util

import pytest


def test_init(ctrl, obs):
    # assert ctrl._db is None
    scheduling_intervals = ctrl._da_scheduler._scheduling_intervals
    target_schedule = ctrl._da_scheduler._target_schedule
    weights = ctrl._da_scheduler._weights
    assert scheduling_intervals > 0
    assert len(target_schedule) == scheduling_intervals
    assert len(weights) == scheduling_intervals


def test_register(ctrl, obs, ua_mocks):
    """Test UA registering at the ctrl."""
    assert len(ctrl._agents) == len(ua_mocks)
    agents = list(sorted(ctrl._agents.items(), key=lambda i: i[1]))
    assert agents[0][0]._path == 'agents/0'
    assert agents[0][1] == 'tcp://127.0.0.1:5556/0'


@pytest.mark.asyncio
def test_update_thermal_demand_forecast(ctrl, obs, ua_mocks, monkeypatch):
    ctrl_proxy = ua_mocks[0].ctrl_agent
    # Note: Controller for testing is initialized with current system time.
    #       *start* has thus to be set in the future of utcnow()
    start = arrow.utcnow().replace(days=1, hour=0, minute=0, second=0,
                                   microsecond=0)
    res = 900
    data = np.arange(96 * 2)
    fc = util.TimeSeries(start, res, data)
    target_schedule = util.TimeSeries(start, res, [0 for i in range(96)])
    weights = util.TimeSeries(start, res, [0 for i in range(96)])

    # Patch ControllerAgent
    @coroutine
    def run_negotiation(neg_start):
        return start

    monkeypatch.setattr(ctrl, '_run_negotiation', run_negotiation)

    # Call method and assert that a new negotiation has been started
    yield from ctrl_proxy.update_thermal_demand_forecast(target_schedule, weights, fc)
    ret = yield from ctrl._task_negotiation
    assert ret == start

    # Assert that all UnitAgents got the (same) data
    futs = [ua.update_forecast_called for ua in ua_mocks]
    ret = yield from gather(*futs)
    for i in ret:
        assert i == fc


@pytest.mark.asyncio
@pytest.mark.parametrize('err_kwargs', [
    {'start': arrow.get(0).replace(days=-1)},  # too old
    {'start': arrow.get().replace(minute=3)},  # does not align
    {'res': 3},  # Wrong resolution
    {'data': np.zeros(10)},  # Too short
])
def test_update_thermal_demand_forecast_argchecks(ctrl, obs, err_kwargs):
    """Check various bad arguments."""
    kwargs = {
        'start': ctrl._da_scheduler._forecast_p.start,
        'res': ctrl._da_scheduler._scheduling_res,
        'data': np.zeros(ctrl._da_scheduler._scheduling_intervals),
    }
    kwargs.update(err_kwargs)
    fc = util.TimeSeries(**kwargs)
    target_schedule = util.TimeSeries(**kwargs)
    weights = util.TimeSeries(**kwargs)
    with pytest.raises(ValueError):
        yield from ctrl.update_thermal_demand_forecast(target_schedule, weights, fc)


@pytest.mark.asyncio
def test_udpate_thermal_demand_forecast_neg_running(ctrl, obs):
    """Raise an error if the forecast is update while a negotiation is
    currently running."""
    ctrl._task_negotiation = asyncio.Future()
    kwargs = {
        'start': ctrl._da_scheduler._forecast_p.start,
        'res': ctrl._da_scheduler._scheduling_res,
        'data': np.zeros(ctrl._da_scheduler._scheduling_intervals),
    }
    fc = util.TimeSeries(**kwargs)
    target_schedule = util.TimeSeries(**kwargs)
    weights = util.TimeSeries(**kwargs)
    with pytest.raises(RuntimeError):
        yield from ctrl.update_thermal_demand_forecast(target_schedule, weights, fc)

    # Try again:
    ctrl._task_negotiation.set_result(None)
    yield from ctrl.update_thermal_demand_forecast(target_schedule, weights, fc)


@pytest.mark.asyncio
def test_get_day_ahead_plan(ctrl_obs):
    ctrl, obs = ctrl_obs
    ctrl._da_scheduler._forecast_p = util.TimeSeries(
        arrow.get('2015-01-01T09:00:00'),
        ctrl._da_scheduler._scheduling_res,
        np.arange(ctrl._da_scheduler._scheduling_intervals))
    ctrl._task_negotiation = asyncio.Future()
    ctrl._task_negotiation.set_result(None)

    # 15h, because its 09:00
    r_start = int(15 * 3600 / ctrl._da_scheduler._scheduling_res)
    r_end = r_start + (24 * 3600 // ctrl._da_scheduler._scheduling_res)
    expected_dap = list(range(r_start, r_end))

    # Assert that the ts returned is the correct sub-range of the forecast:
    dap_start = arrow.get('2015-01-02')
    dap_end = dap_start.replace(days=1)
    dap = yield from ctrl.get_day_ahead_plan(dap_start, dap_end)
    assert list(dap) == expected_dap

    # Assert that the correct target_schedule and weight vector have been set
    exp_len = ctrl._da_scheduler._scheduling_intervals
    pad_len = exp_len - r_end  # Length of 0-padding
    assert len(ctrl._da_scheduler._target_schedule) == exp_len
    assert list(ctrl._da_scheduler._target_schedule.data) == \
        list(range(r_end)) + [0] * pad_len
    assert len(ctrl._da_scheduler._weights) == exp_len
    assert list(ctrl._da_scheduler._weights.data) == [1] * r_end + [0] * pad_len


@pytest.mark.asyncio
def test_get_day_ahead_plan_wrong_start(ctrl, obs):
    ctrl._forecast_start = arrow.get('2015-06-08T09:00:00+02:00').to('utc')
    ctrl._task_negotiation = asyncio.Future()
    ctrl._task_negotiation.set_result(None)
    dap_start = ctrl._forecast_start.replace(days=-1)
    dap_end = ctrl._forecast_start.replace(hours=1)

    with pytest.raises(ValueError):
        yield from ctrl.get_day_ahead_plan(dap_start, dap_end)

# FIXME: move this to observer test
# @pytest.mark.asyncio
# def test_update_stats(ctrl, obs, ua_mocks):
#     ctrl._agents = {a: a.addr for a in ua_mocks[:2]}
#     ctrl._neg_done = asyncio.Future()

#     ctrl.update_stats(ua_mocks[0], 0, 1, 2, 3, 4, True)
#     assert ctrl._agent_msgs == {ua_mocks[0]: (3, 4)}

#     ctrl.update_stats(ua_mocks[1], 0, 0, 0, 0, 0, True)
#     assert ctrl._agent_msgs == {ua_mocks[0]: (3, 4), ua_mocks[1]: (0, 0)}

#     ctrl.update_stats(ua_mocks[0], 0, 1, 2, 4, 4, True)
#     assert ctrl._agent_msgs == {ua_mocks[0]: (4, 4), ua_mocks[1]: (0, 0)}

#     ret = yield from ctrl._neg_done
#     assert ret is True


@pytest.mark.asyncio
def test_run_negotiation(ctrl_obs, ua_mocks):
    """Test a simple DAP from the ctrl's point of view.
    No negotitation at unit agents is performed (mocks used).
    """
    ctrl, obs = ctrl_obs
    obs._termination_detector._num_agents = len(ua_mocks)
    # General config
    ctrl._agents = collections.OrderedDict(
        sorted(ctrl._agents.items(), key=lambda i: i[1]))
    start = arrow.get('2015-01-01')
    res = ctrl._scheduling_res = 15

    ctrl._da_scheduler._forecast_p = util.TimeSeries(start, res, [0] * 4)
    ctrl._da_scheduler._target_schedule = util.TimeSeries(start, res, list(range(4)))
    ctrl._da_scheduler._weights = util.TimeSeries(start, res, [1] * 4)

    ts = list(ctrl._da_scheduler._target_schedule.data)
    weights = list(ctrl._da_scheduler._weights.data)

    # conn = {}

    # t = [0, 1, 2, 3]
    # w = [0, 0, 0, 0]

    neg_task = aiomas.async(ctrl._run_negotiation(start))

    # Test init_negotiation()
    futs = [ua.init_negotiation_called for ua in ua_mocks]
    ret = yield from gather(*futs)

    assert ret == [
        (['tcp://127.0.0.1:5556/1', 'tcp://127.0.0.1:5556/2'], start, res, ts, weights, True),  # NOQA
        (['tcp://127.0.0.1:5556/0', 'tcp://127.0.0.1:5556/2'], start, res, ts, weights, False),  # NOQA
        (['tcp://127.0.0.1:5556/0', 'tcp://127.0.0.1:5556/1'], start, res, ts, weights, False),  # NOQA
    ]

    # UA agents found a solution
    for ua in ua_mocks:
        yield from ua.obs_agent.update_stats(ua, 0, 0, 0, 1, 1, True)

    # Retrieve final solution
    yield from neg_task

    assert ctrl._da_scheduler._forecast_p == util.TimeSeries(start, res, [18, 22, 26])

    # Check solutions sent to each UA
    futs = [ua.stop_negotiation_called for ua in ua_mocks]
    yield from gather(*futs)

    # Sent final solution to observer
    futs = [ua.obs_agent.update_final_cand(ua.solution) for ua in ua_mocks]
    yield from gather(*futs)

    futs = [ua.set_schedule_called for ua in ua_mocks]
    ret = yield from gather(*futs)
    solution = ua_mocks[0].solution
    for i, schedule_id in enumerate(ret):
        assert schedule_id == solution.sids[i]


@pytest.mark.asyncio
def test_run_negotiation_timeout(ctrl, obs, ua_mocks):
    """UAs don't detect termination and we time out."""
    ctrl._agents = collections.OrderedDict(
        sorted(ctrl._agents.items(), key=lambda i: i[1]))
    ctrl._neg_timeout = 0.1
    start = arrow.get('2015-01-01')
    res = ctrl._da_scheduler._scheduling_res = 15
    ctrl._da_scheduler._forecast_p = util.TimeSeries(start, res, [0] * 4)
    ctrl._da_scheduler._target_schedule = util.TimeSeries(start, res, list(range(4)))
    ctrl._da_scheduler._weights = util.TimeSeries(start, res, [1] * 4)

    yield from aiomas.async(ctrl._run_negotiation(start))

    # Check solution returned to the BGA
    assert ctrl._da_scheduler._forecast_p == util.TimeSeries(start, res, [18, 22, 26])


@pytest.mark.parametrize(['n_agents', 'topo'], [
    (1, {0: set()}),
    (2, {0: {'1'}, 1: {'0'}}),
    (8, {
        0: {'1', '4', '7'},
        1: {'0', '2', '4'},
        2: {'1', '3', '5'},
        3: {'2', '4'},
        4: {'0', '1', '3', '5'},
        5: {'2', '4', '6'},
        6: {'5', '7'},
        7: {'0', '6'},
    }),
])
def test_make_topology(ctrl, n_agents, topo):
    ctrl._topology_manager._topology_phi = .5
    ctrl._topology_manager._topology_seed = 23
    ctrl._agents = {i: str(i) for i in range(n_agents)}
    connections = ctrl._topology_manager.make_topology(ctrl._agents)
    assert connections == topo
