"""
Implementation of the `mosaik API`__

__ https://mosaik.readthedocs.org/en/latest/mosaik-api/high-level.html

"""
import io
import json
import lzma
import os.path

import arrow
import mosaik_api
import numpy as np

from chpsim.sim import ChpConfig, CHP


DEFAULT_DATA = os.path.join(os.path.dirname(__file__), 'data', 'demand.csv.xz')


class ChpSim(mosaik_api.Simulator):
    def __init__(self):
        meta = {
            'models': {
                'CHP': {
                    'public': True,
                    'params': [
                        'chp_p_levels',
                        'chp_min_off_time',
                        'storage_e_th_max',
                        'storage_e_th_init',
                    ],
                    'attrs': [
                        'p_th_demand',  # input
                        'chp_p_el',  # in- and output
                        'chp_p_el_levels',
                        'chp_p_th',
                        'chp_p_th_levels',
                        'chp_setpoint',
                        'chp_min_off_time',
                        'chp_off_since',
                        'storage_e_th',
                        'storage_e_th_max',
                    ],
                },
            },
        }
        super().__init__(meta)

        self._demand = None  # Demand file
        self._chps = {}  # Maps EIDs to CHP index
        self._chp_config = []
        self._sim = None  # CHP sim instance

    def init(self, sid, start_date, demand_file=DEFAULT_DATA):
        start_date = arrow.get(start_date).to('utc')

        open = lzma.open if demand_file.endswith('.xz') else io.open
        self._demand = open(demand_file, 'rt')
        demand_meta = json.loads(next(self._demand).strip())
        assert demand_meta['interval_minutes'] == 1
        demand_start = arrow.get(demand_meta['start_time']).to('utc')
        start_diff = int((start_date - demand_start).total_seconds() / 60)
        assert start_diff >= 0
        for i in range(start_diff):
            next(self._demand)

        return self.meta

    def create(self, num, model, **chp_params):
        # Mosaik takes care that we only get valid values for *num* and *model*
        n_chps = len(self._chp_config)
        entities = []
        for i in range(n_chps, n_chps + num):
            eid = 'chp-%s' % i
            self._chps[eid] = i
            self._chp_config.append(ChpConfig(**chp_params))
            entities.append({'eid': eid, 'type': model})

        return entities

    def setup_done(self):
        self._sim = CHP(self._chp_config)

    def step(self, time, inputs):
        data = next(self._demand).strip().split(',')
        p_el, p_th_heat, p_th_water = map(float, data)
        sum_p_th = p_th_heat + p_th_water

        p_th_demand = [sum_p_th] * len(self._chps)
        setpoints = [None] * len(self._chps)

        # set data from *inputs* to chps
        for chp, chp_inputs in inputs.items():
            idx = self._chps[chp]
            if 'chp_p_el' in chp_inputs:
                _, p_el = chp_inputs['chp_p_el'].popitem()
                p_el_levels = self._sim.chp_p_levels[idx,:,0]
                sp = np.argmin(np.abs(p_el_levels - p_el))
                setpoints[idx] = sp

        self._sim.step(p_th_demand, setpoints)

        return time + 60

    def get_data(self, outputs):
        # Make an intermediate Sim instance if static are requested and the
        # acutal simulation has not yet begun:
        sim = CHP(self._chp_config) if self._sim is None else self._sim

        data = {}
        for eid, attrs in outputs.items():
            if eid not in self._chps:
                raise ValueError('Unknown entity ID "%s"' % eid)

            idx = self._chps[eid]
            data[eid] = {}
            for attr in attrs:
                data[eid][attr] = getattr(sim, attr)[idx].tolist()

        return data


def main():
    return mosaik_api.start_simulation(ChpSim(), 'CHP simulator')


if __name__ == '__main__':
    main()

