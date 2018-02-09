import itertools

from aiomas import expose
import numpy as np

from openvpp_agents import util
from openvpp_agents.unit import UnitModel


# The simulation has a temporal resolution of 1 minute.
# The ChpModel needs this value in seconds for computing the interval sice
# for _schedules and forecasts.
# We also need this value to compute an amount of energy [Wh] based on a
# certain power [W] during a simulation step.
SIM_RES = 60  # seconds
SIM_RES_MINUTES = SIM_RES // 60  # minutes
HOUR_MINUTES = 60  # An hour has 60 minutes
WH_FACTOR = 1 / (HOUR_MINUTES / SIM_RES_MINUTES)  # W * WH_FACTOR = Wh / step


class ChpModel(UnitModel):
    def __init__(self, chp_p_levels, chp_min_off_time, storage_e_th_max):
        # Params and hard constraints
        self.chp_p_levels = chp_p_levels
        self.chp_min_off_time = chp_min_off_time
        self.storage_e_th_max = storage_e_th_max

        # Soft constraints
        self.storage_e_th_min = 0.05 * storage_e_th_max  # [Wh]
        # self.min_run_time = 4 * 60  # [min]
        # self.max_starts_per_day = 4

        self.model = CHP(chp_p_levels, chp_min_off_time, storage_e_th_max,
                         self.storage_e_th_min)

        self._forecast_demand_p_th = None
        self._schedule_id = None
        self._schedules = None

    def get_schedule(self, schedule_id):
        return self._schedules[schedule_id]

    @expose
    def update_forecast(self, fc):
        # TODO: unify resolution check?
        if max(fc.res, self.model.res) % min(fc.res, self.model.res) != 0:
            raise ValueError('fc.res=%ss must be a multiple or an integral '
                             'divider of the model resolution %ds' %
                             (fc.res, self.model.res))

        self._forecast_demand_p_th = fc

    def generate_schedules(self, start, res, intervals, state):
        # TODO for electrical target schedule: generate more than one possible
        # schedule / first in a simple manner to verify adaptation to
        # electrical target schedule optimization
        # TODO: unify resolution check?
        # TODO: schedule resolution is 60 (thus a minute)

        if res < self.model.res or res % self.model.res != 0:
            raise ValueError('Schedule resolution must be >= %d and a '
                             'multiple of it' % self.model.res)

        # The latest unit state should be from time "start", but we want
        # the state from the interval before.
        # Example: "start" is 09:00 o'clock, then the last state is for the
        # interval [09:00, 09:01).  If we want to simulate from 09:00, we need
        # the previous state from [08:59, 09:00).
        assert state[-1][0] == start
        state = state[-2][1]

        self._schedules = {}
        self._schedule_id = itertools.count()

        # Start index in the array with the forecast data
        res_ratio = int(res // self.model.res)
        n_sim_steps = res_ratio * intervals

        # Generate initial schedule
        setpoint = self._setpoint_for(state['chp_p_el'])
        storage_e = state['storage_e_th']
        self.model.reset(chp_setpoint=setpoint, storage_e_th=storage_e)
        p_el = np.zeros(n_sim_steps)
        p_th_gen = self._forecast_demand_p_th.iter(
            start, self.model.res, n_sim_steps)
        for i, p_th in enumerate(p_th_gen):
            self.model.step(p_th)
            p_el[i] = self.model.chp_p_el

        schedule_id = next(self._schedule_id)
        unit_schedule = util.TimeSeries(start, self.model.res, p_el)
        utility = 1

        # reshaping to 15 min resolution
        negotiation_schedule = p_el.reshape(intervals, res_ratio).mean(axis=1)

        self._schedules[schedule_id] = unit_schedule
        possible_schedules = [(schedule_id, utility, negotiation_schedule)]

        # TODO: quick hack for testing multiple schedules - remove
        # np.random.seed(7)  # at least .. deterministic hack
        # for i in range(10):
        #     tmp_data = np.random.permutation(unit_schedule.data)
        #     tmp = util.TimeSeries(start, self.model.res, tmp_data)
        #     # print(tmp.data)
        #     sched_id = next(self._schedule_id)
        #     self._schedules[sched_id] = tmp
        #     # reshaping to 15 min resolution
        #     negotiation_schedule = tmp_data.reshape(
        #                                 intervals, res_ratio
        #                                 ).mean(axis=1)
        #     possible_schedules.append((sched_id, utility,
        #                                negotiation_schedule))

        # print("sched 6:")
        # for val in self._schedules[6].data:
        #     print(val)
        # end quick hack

        # return [(schedule_id, utility, negotiation_schedule)]
        return possible_schedules

    def _setpoint_for(self, p_el):
        p_el_levels = np.array([l[0] for l in self.model.chp_p_levels])
        return np.argmin(np.abs(p_el_levels - p_el))


class CHP:
    """CHP simulator."""
    def __init__(self, chp_p_levels, chp_min_off_time, storage_e_th_max,
                 storage_e_th_min):
        self.chp_setpoint = 0
        self.chp_setpoint_max = len(chp_p_levels) - 1

        self.chp_p_levels = chp_p_levels

        self.chp_off_since = chp_min_off_time
        self.chp_min_off_time = chp_min_off_time

        self.storage_e_th = 0
        self.storage_e_th_max = storage_e_th_max
        self.storage_e_th_min = storage_e_th_min

    @property
    def res(self):
        """The simulation resolution in seconds."""
        return SIM_RES

    @property
    def chp_p_el(self):
        return self.chp_p_levels[self.chp_setpoint][0]

    @property
    def chp_p_th(self):
        return self.chp_p_levels[self.chp_setpoint][1]

    def reset(self, chp_setpoint=None, storage_e_th=None):
        if chp_setpoint is not None:
            assert 0 <= chp_setpoint <= self.chp_setpoint_max, chp_setpoint
            if self.chp_setpoint > 0 and chp_setpoint == 0:
                self.chp_off_since = 0
            self.chp_setpoint = chp_setpoint

        if storage_e_th is not None:
            assert 0 <= storage_e_th <= self.storage_e_th_max, storage_e_th
            self.storage_e_th = storage_e_th

    def step(self, p_th_demand, chp_setpoint=None):
        self.check_chp_status()
        self.set_setpoint(chp_setpoint)

        e_diff = (self.chp_p_th - p_th_demand) * WH_FACTOR
        storage_e_new = self.storage_e_th + e_diff

        # Remainder will be covered by the heater (currently not modeled):
        storage_e_new = max(0, storage_e_new)

        self.storage_e_th = storage_e_new

        self.chp_off_since += 1

    def set_setpoint(self, setpoint):
        if setpoint is None:
            # No changes required
            return

        if self.chp_setpoint == 0 and setpoint > 0:
            # CHP is off and should be turned on
            if self.chp_off_since < self.chp_min_off_time:
                # Nope.  Must stay off.
                return
        elif self.chp_setpoint > 0 and setpoint == 0:
            # CHP is on an should be turned off
            self.chp_off_since = 0

        self.chp_setpoint = setpoint

    def check_chp_status(self):
        if self.storage_e_th >= self.storage_e_th_max:
            # Set setpoint to "0" (off) if storage is full.
            self.set_setpoint(0)
        elif (self.storage_e_th <= self.storage_e_th_min and
              self.chp_setpoint == 0):
            # Set setpoint to max. power if storage is empty and chp is off.
            self.set_setpoint(self.chp_setpoint_max)
