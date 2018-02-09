import asyncio
import functools

from aiomas import expose
import arrow
import logging
import numpy as np
import pytest

from openvpp_agents import util
from openvpp_agents import unit
from openvpp_agents import controller


pytestmark = pytest.mark.system

NUM_EXP = 25  # should be set to 100 for statistical evaluation
N_CASES = 1  # The number of test cases to run
ps_set = [
    [[1, 1, 1, 1, 1], [4, 3, 3, 3, 3], [6, 6, 6, 6, 6], [9, 8, 8, 8, 8], [11, 11, 11, 11, 11]],  # NOQA
    [[13, 12, 12, 12, 12], [15, 15, 15, 14, 14], [18, 17, 17, 17, 17], [20, 20, 20, 19, 19], [23, 22, 22, 22, 22]],  # NOQA
    [[25, 24, 23, 23, 23], [27, 26, 26, 25, 25], [30, 29, 28, 28, 28], [32, 31, 31, 30, 30],    [35, 34, 33, 33, 33]],  # NOQA
    [[36, 35, 35, 34, 34], [39, 38, 37, 36, 36], [41, 40, 40, 39, 39], [44, 43, 42, 41, 41], [46, 45, 45, 44, 44]],  # NOQA
    [[48, 47, 46, 45, 45], [50, 49, 48, 48, 47], [53, 52, 51, 50, 50], [55, 54, 53, 53, 52], [58, 57, 56, 55, 55]],  # NOQA
    [[60, 58, 57, 56, 56], [62, 61, 60, 59, 58], [65, 63, 62, 61, 61], [67, 66, 65, 64, 63], [70, 68, 67, 66, 66]],  # NOQA
    [[71, 70, 68, 67, 67], [74, 72, 71, 70, 69], [76, 75, 73, 72, 72], [79, 77, 76, 75, 74], [81, 80, 78, 77, 77]],  # NOQA
    [[83, 81, 80, 78, 78], [85, 83, 82, 81, 80], [88, 86, 85, 83, 83], [90, 88, 87, 86, 85], [93, 91, 90, 88, 88]],  # NOQA
    [[95, 92, 91, 90, 89], [97, 95, 93, 92, 91], [100, 97, 96, 95, 94], [102, 100, 98, 97, 96], [105, 102, 101, 100, 99]],  # NOQA
    [[106, 104, 102, 101, 100], [109, 106, 105, 103, 102], [111, 109, 107, 106, 105], [114, 111, 110, 108, 107], [116, 114, 112, 111, 110]]  # NOQA
]


class UnitModelStub(unit.UnitModel):
    def __init__(self, ps):
        self._ps = ps

    @expose
    def update_forecast(self, fc):
        pass

    def generate_schedules(self, start, res, intervals, state):
        return [(i, 1., os) for i, os in enumerate(self._ps)]


class UnitIfStub(unit.UnitInterface):
    def __init__(self, *args, **kwargs):
        pass

    @property
    def state(self):
        return 42

    @expose
    def set_schedule(self, *args, **kwargs):
        pass


@pytest.yield_fixture
def uas(containers, ctrl_obs, event_loop):
    ctrl, obs = ctrl_obs
    uas = []
    for i in range(10):
        ps = ps_set[i]
        ua_fut = unit.UnitAgent.factory(
            containers[1],
            ctrl_agent_addr=ctrl.addr,
            obs_agent_addr=obs.addr,
            unit_model=(__name__ + ':UnitModelStub', {'ps': ps}),
            unit_if=(__name__ + ':UnitIfStub', {}),
            planner=('openvpp_agents.planning:Planner',
                     {'check_inbox_interval': (0.01, 0.02)}),
            sleep_before_connect=False)
        ua = event_loop.run_until_complete(ua_fut)
        uas.append(ua)
    yield uas
    [u.stop() for u in uas]


@pytest.yield_fixture
def ctrl(containers):
    ctrl = controller.ControllerAgent(containers[0], scheduling_period=5*900)
    yield ctrl
    ctrl.stop()


@pytest.mark.parametrize('target', [
    np.array([538, 524, 515, 507, 505]),
    np.array([539, 525, 516, 508, 506]),
    np.array([540, 526, 517, 509, 507]),
    np.array([541, 527, 518, 510, 508]),
    np.array([542, 528, 519, 511, 509]),
    np.array([543, 529, 520, 512, 510]),
    np.array([544, 530, 521, 513, 511]),
    np.array([545, 531, 522, 514, 512]),
    np.array([546, 532, 523, 515, 513]),
    np.array([547, 533, 524, 516, 514]),
    np.array([548, 534, 525, 517, 515]),
    np.array([549, 535, 526, 518, 516]),
    np.array([550, 536, 527, 519, 517]),
    np.array([551, 537, 528, 520, 518]),
    np.array([552, 538, 529, 521, 519]),
    np.array([553, 539, 530, 522, 520]),
    np.array([554, 540, 531, 523, 521]),
    np.array([555, 541, 532, 524, 522]),
    np.array([556, 542, 533, 525, 523]),
    np.array([557, 543, 534, 526, 524]),
    np.array([558, 544, 535, 527, 525]),
    np.array([559, 545, 536, 528, 526]),
    np.array([560, 546, 537, 529, 527]),
    np.array([561, 547, 538, 530, 528]),
    np.array([562, 548, 539, 531, 529]),
    np.array([563, 549, 540, 532, 530]),
    np.array([564, 550, 541, 533, 531]),
    np.array([565, 551, 542, 534, 532]),
    np.array([566, 552, 543, 535, 533]),
    np.array([567, 553, 544, 536, 534]),
    np.array([568, 554, 545, 537, 535]),
    np.array([569, 555, 546, 538, 536]),
    np.array([570, 556, 547, 539, 537]),
    np.array([571, 557, 548, 540, 538]),
    np.array([572, 558, 549, 541, 539]),
    np.array([573, 559, 550, 542, 540]),
    np.array([574, 560, 551, 543, 541]),
    np.array([575, 561, 552, 544, 542]),
    np.array([576, 562, 553, 545, 543]),
    np.array([577, 563, 554, 546, 544]),
    np.array([578, 564, 555, 547, 545]),
    np.array([579, 565, 556, 548, 546]),
    np.array([580, 566, 557, 549, 547]),
    np.array([581, 567, 558, 550, 548]),
    np.array([582, 568, 559, 551, 549]),
    np.array([583, 569, 560, 552, 550]),
    np.array([584, 570, 561, 553, 551]),
    np.array([585, 571, 562, 554, 552]),
    np.array([586, 572, 563, 555, 553]),
    np.array([587, 573, 564, 556, 554]),
    np.array([588, 574, 565, 557, 555]),
    np.array([589, 575, 566, 558, 556]),
    np.array([590, 576, 567, 559, 557]),
    np.array([591, 577, 568, 560, 558]),
    np.array([592, 578, 569, 561, 559]),
    np.array([593, 579, 570, 562, 560]),
    np.array([594, 580, 571, 563, 561]),
    np.array([595, 581, 572, 564, 562]),
    np.array([596, 582, 573, 565, 563]),
    np.array([597, 583, 574, 566, 564]),
    np.array([598, 584, 575, 567, 565]),
    np.array([599, 585, 576, 568, 566]),
    np.array([600, 586, 577, 569, 567]),
    np.array([601, 587, 578, 570, 568]),
    np.array([602, 588, 579, 571, 569]),
    np.array([603, 589, 580, 572, 570]),
    np.array([604, 590, 581, 573, 571]),
    np.array([605, 591, 582, 574, 572]),
    np.array([606, 592, 583, 575, 573]),
    np.array([607, 593, 584, 576, 574]),
    np.array([608, 594, 585, 577, 575]),
    np.array([609, 595, 586, 578, 576]),
    np.array([610, 596, 587, 579, 577]),
    np.array([611, 597, 588, 580, 578]),
    np.array([612, 598, 589, 581, 579]),
    np.array([613, 599, 590, 582, 580]),
    np.array([614, 600, 591, 583, 581]),
    np.array([615, 601, 592, 584, 582]),
    np.array([616, 602, 593, 585, 583]),
    np.array([617, 603, 594, 586, 584]),
    np.array([618, 604, 595, 587, 585]),
    np.array([619, 605, 596, 588, 586]),
    np.array([620, 606, 597, 589, 587]),
    np.array([621, 607, 598, 590, 588]),
    np.array([622, 608, 599, 591, 589]),
    np.array([623, 609, 600, 592, 590]),
    np.array([624, 610, 601, 593, 591]),
    np.array([625, 611, 602, 594, 592]),
    np.array([626, 612, 603, 595, 593]),
    np.array([627, 613, 604, 596, 594]),
    np.array([628, 614, 605, 597, 595]),
    np.array([629, 615, 606, 598, 596]),
    np.array([630, 616, 607, 599, 597]),
    np.array([631, 617, 608, 600, 598]),
    np.array([632, 618, 609, 601, 599]),
    np.array([633, 619, 610, 602, 600]),
    np.array([634, 620, 611, 603, 601]),
    np.array([635, 621, 612, 604, 602]),
    np.array([636, 622, 613, 605, 603]),
    np.array([637, 623, 614, 606, 604]),
][:N_CASES])
@pytest.mark.asyncio
def test_dummy(uas, ctrl_obs, target):
    # logging.basicConfig(level=logging.DEBUG)
    ctrl, obs = ctrl_obs
    results = []
    msgs = []
    for i in range(NUM_EXP):
        start = arrow.get().replace(minute=0, second=0, microsecond=0)
        ctrl._target_schedule = util.TimeSeries(start, 900, target)
        ctrl._weights = util.TimeSeries(start, 900, [1] * 5)
        ctrl._forecast_p = util.TimeSeries(start, 900, [0]*5)
        ctrl._scheduling_period = 900 * 5

        target_schedule = util.TimeSeries(start, 900, [0] * 5)
        weights = util.TimeSeries(start, 900, [0] * 5)
        fc = util.TimeSeries(start, 900, [0] * 5)
        yield from ctrl.update_thermal_demand_forecast(target_schedule,
            weights, fc)
        dap = yield from ctrl.get_day_ahead_plan(
            start, start.replace(seconds=5*900))

        diff_abs = np.abs(dap - target)
        diff_mean = np.mean(diff_abs / target)
        assert diff_mean <= 0.02
        results.append(diff_mean)

    assert np.mean(results) <= 0.05
