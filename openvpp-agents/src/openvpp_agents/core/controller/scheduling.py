import asyncio
import logging
from asyncio import coroutine

import numpy as np

from openvpp_agents import util

logger = logging.getLogger(__name__)


class DayAheadScheduler:

    def __init__(self, controller, scheduling_res, scheduling_period):
        self._controller = controller
        self._scheduling_res = scheduling_res
        self._scheduling_period = scheduling_period
        self._scheduling_intervals = int(scheduling_period / scheduling_res)
        # We need a basic TimeSeries as long as we have no real data.
        # We take utc 7a.m., as if the schedule were the result of a dap
        utctime = self._controller.container.clock.utcnow()
        utctime = utctime.replace(minute=0, second=0, microsecond=0, hour=7)
        zero_ts = util.TimeSeries(utctime,
                                  self._scheduling_res,
                                  np.zeros(self._scheduling_intervals))
        # TODO wording: Maybe this should be called "planned_p", because its
        # the result of a negotiation – the aggregate power output of the VPP:
        self._forecast_p = zero_ts
        self._target_schedule = zero_ts.copy()
        # The *weights* list allows us to express that intervals closer to
        # "now" must match the target schedule closer than intervals farther in
        # the future:
        self._weights = zero_ts.copy()

        self._start = None

    @coroutine
    def update_thermal_demand_forecast(self, target_schedule, weights, fc):
        """Receive a new thermal demand forecast from an external source and
        forward it to all agents.  When done, start a new negotiation.
        The thermal demand forecast has to be scaled to one CHP's demand and is
        not specific for individual agents. Adapt this using an agent-CHP
        mapping if needed.

        Raise a :exc:`ValueError` or :exc:`RuntimeError` if the forecast has
        not the proper size or start date.

        Called by external parties (like the mosaik agent).

        """
        # FIXME: Comments are not correct below
        # This method is a stub that demonstrates how external forecast data
        # can be received and passed to the agents.
        # wait for the observer to be registered before starting negotiation

        yield from self._controller._observer_registered
        yield from self._controller._agents_registered

        util.check_date_diff(fc.start, self._forecast_p.start,
                             self._scheduling_res)

        if fc.period != self._scheduling_period:
            raise ValueError('The provided data and resolution (%d) do '
                             'not match the scheduling period (%d).'
                             % (fc.period, self._scheduling_period))

        if (self._controller._task_negotiation is not None and
                not self._controller._task_negotiation.done()):
            raise RuntimeError('Cannot set new forecast.  A negotiation is '
                               'still in progress.')

        futs = [a.update_forecast(fc) for a in self._controller._agents]
        yield from asyncio.gather(*futs)

        # we have to append the new target schedule to the old one and
        # set the weight vector accordingly
        self._target_schedule.extend(target_schedule)
        self._weights.extend(weights)

    def init_negotiation(self, start):
        # TODO: Add test for possible hidden bug.
        self._target_schedule.shift(start)
        self._weights.shift(start)
        self._start = start

        res = self._target_schedule.res
        ts = self._target_schedule.data
        weights = self._weights.data
        return res, ts, weights

    def finalize_negotiation(self, solution):
        # TODO wording: forecast_p = aggregierter Einsatzplan
        forecast_p = self._aggregate_solution(solution)
        assert self._start is not None
        res = self._target_schedule.res
        self._forecast_p = util.TimeSeries(self._start, res, forecast_p)

    def _aggregate_solution(self, solution):
        """Return the the element-wise sum of all schedules."""
        return solution.cs.sum(axis=0)

    def get_day_ahead_plan(self, dap_start, dap_end):
        """Return the day ahead plan (which is the result of a negotiation).

        """
        fc = self._forecast_p

        if dap_start < fc.start:
            raise ValueError('dap_start must be >= the forecast start')
        if dap_end > fc.end:
            raise ValueError('dap_end must be <= the forecast end')

        # TODO FIXME: is this still correct?
        # FIXME cont: Maybe _forecast_p should be handled different?
        #
        # Update target_schedule and weights array.
        # Because we communicate our planned generation to an external party,
        # we have to set it as new target schedule for our VPP:
        self._target_schedule = util.TimeSeries(fc.start, fc.res,
                                                np.zeros(len(fc)))
        self._target_schedule[:dap_end] = fc[:dap_end]
        self._weights = util.TimeSeries(fc.start, fc.res, np.zeros(len(fc)))
        self._weights[:dap_end] = 1

        # Extract DAP
        dap = fc[dap_start:dap_end]

        return dap
