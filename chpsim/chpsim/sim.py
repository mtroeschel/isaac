"""
Simple implementation of a cobined heat and power plant (CHP).  The
flow-diagram belows shows how our model roughly works::

              +------------------------------------------+
              |                      +-------------+     |
              |  +------------------>|   Heater    |--+  |
              |  |                   +-------------+  |  |
              |  |  +-------------+                   |  |
   Setpoint ->|--|->| Controller  |                   +->|--> Heat
              |  |  +-------------+                   |  |
        Gas ->|--+       |           +-------------+  |  |
              |  |       |   +------>| HeatStorage |--+  |
              |  |       |   | P_th  +-------------+     |
              |  |       v   |                           |
              |  |  +-------------+ P_el                 |
              |  +->|     CHP     |--------------------->|--> el. Power
              |     +-------------+                      |
              +------------------------------------------+

The *CHP* consist of an engine, a generator and a heat exchanger.  It burns gas
and converts its energy into thermal and electrical power.  The CHP can be
driven at different *setpoints* – at least *off* and *max. power (el., th.)*
(both in [W]).  The quotient *p_el_max* / *p_th_max* is called the *CHP
coefficient*.  If a CHP is turned off, it usually has to stay offline for
a while (*min_off_time* in [min]).

The thermal power is stored in a *HeatStorage* (a cistern) with a maximum
storage capacity *e_th_max* in [Wh].  The current capacity *e_th* is also
measured in [Wh].  It also constanly looses some heat with a rate of
*heat_loss*, in [W].  Its max. thermal power output *p_th_max* is measured in
[W].

A *Heater* backs up the *HeatStorage* and burns gas in order to heat water.

The *Controller* turns on the CHP if the heat storage is empty and turns it of
if it is full.  It may also receive a *setpoint* for the CHP from an external
source.

The model implemented so far is even simpler.  We don't yet model the heater,
storage heat loss and the storage's max. thermal power output.

The :class:`CHP` class doesn't model a single CHP, but a list of them using
NumPy arrays for performance reasons.  It gets configured with a list of
:class:`ChpConfig` instances which represent single CHPs.

"""
from collections import namedtuple

import numpy as np


HOUR = 60  # An hour has 60 minutes
STEP_MINUTES = 1


ChpConfig = namedtuple('ChpConfig', ['chp_p_levels',
                                     'chp_min_off_time',
                                     'storage_e_th_max',
                                     'storage_e_th_init'])


class CHP:
    """Simulator for a number of CHPs.

    The list *chp_config* contains one :class:`ChpConfig` instances for each
    CHP to be simulated.

    """
    def __init__(self, chp_config):
        # "chp_config" contains the data row-wise (one CHP per entry), but
        # we need it column-wise (one array for each attribute).

        self.chp_setpoint = np.zeros(len(chp_config), int)
        """Setpoints for CHP engines.  Initially, they are ``0`` (off)."""

        self.chp_setpoint_max = len(chp_config[0].chp_p_levels) - 1
        """Max. setpoint for all CHPs."""

        chp_p_levels = [chp.chp_p_levels for chp in chp_config]
        self.chp_p_levels = np.array(chp_p_levels)
        """List of possible power levels of all CHPs, including 0 and *P_max*.
        """

        # It's important to make a copy of the P_el slice:
        self.chp_p_el = self.chp_p_levels[:, 0, 0].copy()
        """Current active power output of all CHPs."""

        # It's important to make a copy of the P_th slice:
        self.chp_p_th = self.chp_p_levels[:, 0, 1].copy()
        """Current thermal power output of all CHPs."""

        chp_min_off_time = [chp.chp_min_off_time for chp in chp_config]
        self.chp_min_off_time = np.array(chp_min_off_time, int)
        """Minimum time in minutes the CHPs have to stay turned off."""

        self.chp_off_since = np.array(chp_min_off_time, int)
        """Time (in minutes) passed since the CHPs were turned off."""

        storage_e_th_max = [chp.storage_e_th_max for chp in chp_config]
        self.storage_e_th_max = np.array(storage_e_th_max)
        """Maximum energy level of the storage."""

        storage_e_th_init = [chp.storage_e_th_init for chp in chp_config]
        self.storage_e_th = np.array(storage_e_th_init)
        """Current energy level of the storage."""

    def step(self, p_th_demand, chp_setpoint=None):
        """Make a simulation step of one minute.

        *p_th_demand* is a list of thermal demand values for each CHP.

        *chp_setpoint* is an optional list of setpoints for all CHPs.

        """
        self.check_chp_status()
        if chp_setpoint:
            self.set_setpoint(chp_setpoint)

        e_diff = (self.chp_p_th - p_th_demand) / HOUR
        storage_e_new = self.storage_e_th + e_diff

        # Remainder will be covered by the heater (currently not modeled):
        storage_e_new[storage_e_new < 0] = 0

        self.storage_e_th = storage_e_new

        self.chp_off_since += 1

    def set_setpoint(self, setpoints):
        """Set a list of new *setpoints* to all CHPs.

        The list entries should be valid power levels for the CHP or ``None``
        to not chane the current setpoint.

        """
        assert len(setpoints) == len(self.chp_setpoint)
        for i, setpoint in enumerate(setpoints):
            if setpoint is None:
                # No changes required
                continue

            if self.chp_setpoint[i] == 0 and setpoint > 0:
                # CHP is off and should be turned on
                if self.chp_off_since[i] < self.chp_min_off_time[i]:
                    # Nope.  Must stay off.
                    continue
            elif self.chp_setpoint[i] > 0 and setpoint == 0:
                # CHP is on an should be turned off
                self.chp_off_since[i] = 0

            self.chp_setpoint[i] = setpoint
            p_el, p_th = self.chp_p_levels[i, setpoint]
            self.chp_p_el[i] = p_el
            self.chp_p_th[i] = p_th

    def check_chp_status(self):
        """Check current storage capacity and decided if CHP needs to be
        turned on or off."""
        # Array of "None" -> no setpoint changes
        setpoints = np.full(len(self.chp_setpoint), None, object)

        # Set setpoint to "0" (off) if storage is full.
        where_full = self.storage_e_th >= self.storage_e_th_max
        setpoints[where_full] = 0

        # Set setpoint to max. power if storage is empty and chp is off.
        where_empty = (self.storage_e_th == 0) * (self.chp_setpoint == 0)
        setpoints[where_empty] = self.chp_setpoint_max

        self.set_setpoint(setpoints)
