import json
import math

from openvpp_agents import util
import arrow
import h5py
import numpy as np

from scenario import START_DATE

gray_logo = 'hsl(189, 3%, 46%)'
green_logo = 'hsl(65, 100%, 40%)'
green_darker = 'hsl(67, 75%, 38%)'
green_dark = 'hsl(68, 73%, 35%)'
purple = 'hsl(281, 61%, 40%)'

_cache = {}

TZ = 'Europe/Berlin'


def n(agent):
    agent = agent.decode()
    agent = agent.split('://')[-1]  # Remove the "tcp://" prefix
    agent = agent.replace('/', '-').replace(':', '-')
    return agent


def steps(data):
    for i, d in enumerate(data):
        yield (i, d)
        yield (i + 1, d)


def make_curve(data):
    return [{'x': x, 'y': y} for x, y in steps(data)]


def get_normalized_bounds(*ds):
    min_ = float(min(x[0][0] for x in ds))
    max_ = float(max(x[-1][0] for x in ds))
    diff, min_, max_ = min_, 0, max_ - min_
    return min_, max_, diff


def get_agents(group):
    if 'agents' not in _cache:
        _cache['agents'] = [n(a) for a in group['agents']]
    return _cache['agents']


def get_connections(group):
    if 'connections' not in _cache:
        _cache['connections'] = [(n(s), n(t)) for s, t in group['topology']]

    return _cache['connections']


def get_perf_data(dataset, t_diff):
    cpu_data = []
    cpu_max = 0
    mem_data = []
    mem_max = 0
    for row in dataset:
        t, cpu, mem = [float(i) for i in row]
        t -= t_diff
        cpu_max = max(cpu_max, cpu)
        mem /= 1024**3  # Convert from B to GiB
        mem_max = max(mem_max, mem)
        cpu_data.append({'x': t, 'y': cpu})
        mem_data.append({'x': t, 'y': mem})

    print('Time (min:sec): %d:%02d' % (t // 60, t % 60))
    print('Max. mem (GiB): %.2f' % mem_max)

    return cpu_data, cpu_max, mem_data, mem_max


def get_dap_data(dataset, t_diff):
    perf_data = []
    perf_min = float('inf')
    perf_max = float('-inf')
    msgs_out_cnt = {}
    msgs_in_cnt = {}
    last_t = None
    sorted_ds = sorted(dataset, key=lambda x: [x['t'], x['agent']])
    for t, agent, perf, complete, msgs_out, msgs_in, _ in sorted_ds:
        agent = n(agent)
        t = float(t) - t_diff
        msgs_out, msgs_in = int(msgs_out), int(msgs_in)
        perf = float(perf)

        perf_min = min(perf_min, perf)
        perf_max = max(perf_max, perf)
        perf_data.append({'x': t, 'y': perf,
                          'class': 'green' if complete else 'gray'})

        t_fixed = int(math.ceil(t * 10))  # deci seconds
        if last_t is None:
            # Initialize message count with empty dicts
            last_t = t_fixed - 1
            msgs_out_cnt[last_t] = {}
            msgs_in_cnt[last_t] = {}

        while t_fixed > last_t:
            msgs_out_cnt[last_t + 1] = dict(msgs_out_cnt[last_t])
            msgs_in_cnt[last_t + 1] = dict(msgs_in_cnt[last_t])
            last_t += 1

        msgs_out_cnt[t_fixed][agent] = msgs_out
        msgs_in_cnt[t_fixed][agent] = msgs_in

    # The key *k* is in deci seconds, don't forget to convert it back to sec.
    msgs_out = [{'x': k/10, 'y': sum(v.values())}
                for k, v in sorted(msgs_out_cnt.items())]
    msgs_in = [{'x': k/10, 'y': sum(v.values())}
               for k, v in sorted(msgs_in_cnt.items())]
    msgs_max = msgs_out[-1]['y']

    print('Messages sent: %d' % msgs_max)

    return (perf_data, perf_min, perf_max,
            msgs_out, msgs_in, msgs_max)


def get_sim_data(*dbs):
    sim_start = arrow.get(START_DATE).to(TZ)
    start = util.get_intervals(sim_start.to('utc'),
                               sim_start.replace(days=1).to('utc'), 60)
    end = util.get_intervals(sim_start.to('utc'),
                             sim_start.replace(days=2).to('utc'), 60)
    n_intervals = end - start
    sim_data = []
    p_max = 0
    for db in dbs:
        values = np.zeros(end - start)
        for gn, gd in db['Series'].items():
            if not gn.startswith('ChpSim-0'):
                continue
            values += gd['chp_p_el'][start:end]

        values /= 1000  # Convert from W to kW
        p_max = max(p_max, values.max())
        sim_data.append(make_curve(values))

    return sim_data, n_intervals, p_max


def get_topology(db):
    agents = get_agents(db)
    connections = get_connections(db)
    dataset = db['dap_data'][:]

    t_min, t_max, t_diff = get_normalized_bounds(dataset)
    con_dict = {}
    for a, b in connections:
        con_dict.setdefault(a, set()).add(b)
        con_dict.setdefault(b, set()).add(a)

    perf_min = float('inf')
    perf_max = float('-inf')
    msg_data = []
    mid = 0
    for t, agent, perf, complete, _, _, msgs_sent in dataset:
        # if not msgs_sent:
        #     continue

        agent = n(agent)
        t = float(t) - t_diff
        perf = float(perf)

        perf_min = min(perf_min, perf)
        perf_max = max(perf_max, perf)

        msgs = []
        if msgs_sent:
            for d in con_dict[agent]:
                msgs.append({'id': mid, 's': agent, 'd': d})
                mid += 1

        msg_data.append({
            't': t,
            'perf': perf,
            'class': 'green' if complete else 'gray',
            'msgs': msgs,
        })

    # connections
    # msg_data
    ai = {a: i for i, a in enumerate(agents)}
    topo_data = {
        'nodes': [{'name': a} for a in agents],
        'links': [{
            'source': ai[i],
            'target': ai[j],
            'length': 20 if ai[j] - ai[i] in (1, len(ai) - 1) else 500,
            'strength': 1 if ai[j] - ai[i] in (1, len(ai) - 1) else 0,
        } for i, j in connections],
        'msg_data': msg_data,
        'x_range': [t_min, t_max],
        'y_range': [perf_min, perf_max],
    }
    overlay_topo = {
        'title': 'Overlay topology',
        'type': 'topology',
        'data': topo_data,
    }

    return {
        'legend': False,
        'x_axis': {},
        'y_axis': {},
        'item_names': ['overlay_topo'],
        'enabled': ['overlay_topo'],
        'items': {
            'overlay_topo': overlay_topo,
        }
    }


def get_dap_results(db_agents, db_ctrl, db_nctrl):
    agents = get_agents(db_agents)
    # n_intervals
    # p_max
    cs = db_agents['cs'][:] / 1000  # Convert from W to kW
    ts = db_agents['ts'][:] / 1000  # Convert from W to kW
    fc = db_agents['dap'][:] / 1000  # Convert from W to kW
    # Build stack data with base lines
    base_values = cs.cumsum(axis=0) - cs
    dap_stack_data = [[
        {'v': v, 'y0': y0} for v, y0 in zip(row_cs, row_y0)
    ] for row_cs, row_y0 in zip(cs, base_values)]

    n_intervals = len(ts)
    p_max = max(ts.max(), cs.sum(axis=0).max())

    (data_ctrl, data_nctrl), sim_n_intervals, sim_p_max = get_sim_data(
        db_ctrl, db_nctrl)
    p_max = max(p_max, sim_p_max)

    dap_results = {
        'title': 'DAP results',
        'type': 'stacked_bars',
        'color_range': [green_dark, green_logo],
        'domain_x': [0, n_intervals],
        'domain_y': [0, p_max],
        'data': dap_stack_data,
        'n_layers': len(cs),  # number of agents
        'tooltips': ['Agent %s' % a for a in agents],
    }
    target_schedule = {
        'title': 'Target schedule',
        'type': 'step_curve',
        'class': 'purple',
        'domain_x': [0, n_intervals],
        'domain_y': [0, p_max],
        'data': make_curve(ts),
    }
    forecast = {
        'title': 'Forecast',
        'type': 'step_curve',
        'class': 'gray',
        'domain_x': [0, n_intervals],
        'domain_y': [0, p_max],
        'data': make_curve(fc),
    }
    actual_controlled = {
        'title': 'Actual (controlled sim.)',
        'type': 'step_curve',
        'class': 'blue_dark',
        'domain_x': [0, sim_n_intervals],
        'domain_y': [0, p_max],
        'data': data_ctrl,
    }
    actual_uncontrolled = {
        'title': 'Actual (uncontrolled sim.)',
        'type': 'step_curve',
        'class': 'blue_light',
        'domain_x': [0, sim_n_intervals],
        'domain_y': [0, p_max],
        'data': data_nctrl,
    }

    return {
        'legend': True,
        'x_axis': {
            'range': [0, n_intervals],
            'label': '15 min. intervals',
        },
        'y_axis': {
            'range': [0, p_max],
            'label': 'P',
            'unit': 'kW',
        },
        'item_names': ['dap_results', 'target_schedule', 'forecast',
                       'actual_controlled', 'actual_uncontrolled'],
        'enabled': ['dap_results', 'target_schedule', 'actual_controlled'],
        'items': {
            'dap_results': dap_results,
            'target_schedule': target_schedule,
            'forecast': forecast,
            'actual_controlled': actual_controlled,
            'actual_uncontrolled': actual_uncontrolled,
        },
    }


def get_dap_metrics(db):
    ds_perf = db['perf_data'][:]
    ds_dap = db['dap_data'][:]

    t_min, t_max, t_diff = get_normalized_bounds(ds_perf, ds_dap)

    cpu_data, cpu_max, mem_data, mem_max = get_perf_data(ds_perf, t_diff)
    perf_data, perf_min, perf_max, msgs_out, msgs_in, msgs_max = \
        get_dap_data(ds_dap, t_diff)

    cpu_usage = {
        'title': 'CPU load [%]',
        'type': 'line',
        'class': 'gray',
        'domain_x': [t_min, t_max],
        'domain_y': [0, cpu_max],
        'data': cpu_data,
    }
    mem_usage = {
        'title': 'Memory usage [GiB]',
        'type': 'line',
        'class': 'purple',
        'domain_x': [t_min, t_max],
        'domain_y': [0, mem_max],
        'data': mem_data,
    }
    dap_perf = {
        'title': 'Candidate performance',
        'type': 'scatter',
        'domain_x': [t_min, t_max],
        'domain_y': [perf_min, perf_max],
        'data': perf_data,
    }
    msgs_out = {
        'title': 'Messages sent',
        'type': 'line',
        'class': 'blue_dark',
        'domain_x': [t_min, t_max],
        'domain_y': [0, msgs_max],
        'data': msgs_out,
    }
    msgs_in = {
        'title': 'Messages received',
        'type': 'line',
        'class': 'blue_light',
        'domain_x': [t_min, t_max],
        'domain_y': [0, msgs_max],
        'data': msgs_in,
    }

    return {
        'legend': True,
        'x_axis': {
            'range': [t_min, t_max],
            'label': 'time',
            'unit': 's',
        },
        'y_axis': {
            'range': [0, 1],
            'label': 'relative value',
        },
        'item_names': ['cpu_usage', 'mem_usage', 'dap_perf',
                       'msgs_out', 'msgs_in'],
        'enabled': ['dap_perf', 'mem_usage'],
        'items': {
            'cpu_usage': cpu_usage,
            'mem_usage': mem_usage,
            'dap_perf': dap_perf,
            'msgs_out': msgs_out,
            'msgs_in': msgs_in,
        }
    }


def main():
    dbs = [h5py.File(fname, 'r') for fname in [
        './openvpp-agents.hdf5',
        './results_controlled.hdf5',
        './results_uncontrolled.hdf5',
    ]]
    db_agents, db_ctrl, db_nctrl = dbs
    title, g_dap = list(db_agents['/dap'].items())[0]

    data = {
        'title': title,
        'topology': get_topology(g_dap),
        'dap-results': get_dap_results(g_dap, db_ctrl, db_nctrl),
        'dap-metrics': get_dap_metrics(g_dap),
    }

    with open('html/media/data.js', 'w') as of:
        of.write('var data = %s\n' % json.dumps(data))

    [db.close() for db in dbs]


if __name__ == '__main__':
    main()
