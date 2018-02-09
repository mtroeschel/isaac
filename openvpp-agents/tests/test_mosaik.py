from asyncio import coroutine
import json
import lzma

import arrow
import dateutil.tz
import pytest

from openvpp_agents import mosaik
from openvpp_agents import util


demand_meta = {
    'start_time': '2009-12-31T23:00:00+00:00',
    'interval_minutes': 60,
    'cols': ['p_el', 'p_th_heat', 'p_th_water'],
}
demand_data = """{meta}
0,0,0
0,15,10
0,30,30
""".format(meta=json.dumps(demand_meta))

target_meta = {
    'start_time': '2009-12-31T23:00:00+00:00',
    'interval_minutes': 60,
    'cols': ['p_el', 'weight'],
}
target_data = """{meta}
0,1
0,1
0,1
""".format(meta=json.dumps(target_meta))


@pytest.fixture(params=['csv', 'csv.xz'])
def demand_file(request, tmpdir):
    """Return the path (as string) to either a "demand.csv" or a
    "demand.csv.xz"."""
    ext = request.param
    df = tmpdir.join('test_data.%s' % ext)
    data = {
        'csv': lambda d: d.encode(),
        'csv.xz': lambda d: lzma.compress(d.encode()),
    }
    df.write_binary(data[ext](demand_data))
    return df.strpath


@pytest.fixture
def target_dir(tmpdir):
    tfd = tmpdir.mkdir('targets')
    tf = tfd.join('electricaltarget1.csv')
    tf.write(target_data)
    return tfd.strpath


@pytest.yield_fixture
def user(containers, demand_file, target_dir, monkeypatch):
    # Mock the UserAgent's run() method to avoid nasty side effects:
    monkeypatch.setattr(mosaik.UserAgent, 'run', coroutine(lambda self: None))
    user = mosaik.UserAgent(containers[0], dca_addr=None,
                            demand_file=demand_file,
                            target_dir=target_dir)
    yield user
    user.stop()


def test_user_init_demand_gen(user):
    """Test that the "thermal demand" generator is correctly initialized."""
    ret = user._demand_gen
    assert ret[:2] == (
        arrow.get(demand_meta['start_time']),
        demand_meta['interval_minutes'] * 60,
    )
    assert list(ret[2]) == ['0,0,0\n', '0,15,10\n', '0,30,30\n']


def test_user_get_thermal_demand_forecast(user):
    start = arrow.get(demand_meta['start_time'])
    fc = user._get_thermal_demand_forecast(start, 2)
    assert fc == util.TimeSeries(start, demand_meta['interval_minutes'] * 60,
                                 [0, 25])

    start = start.replace(hours=1)
    fc = user._get_thermal_demand_forecast(start, 2)
    assert fc == util.TimeSeries(start, demand_meta['interval_minutes'] * 60,
                                 [25, 60])


@pytest.mark.parametrize('now, hour, minute, expected', [
    ('2015-01-01T00:00:00', 1, 30, '2015-01-01T00:30:00'),  # today
    ('2015-01-01T12:00:00', 1, 30, '2015-01-02T00:30:00'),  # tomorrow
    ('2015-01-01T00:30:00', 1, 30, '2015-01-02T00:30:00'),  # now -> tomorrow
    ('2015-03-29T00:00:00', 4, 30, '2015-03-29T02:30:00'),  # switch to DST
    ('2015-10-25T00:00:00', 4, 30, '2015-10-25T03:30:00'),  # switch from DST
])
def test_user_get_next_date(user, now, hour, minute, expected):
    now = arrow.get(now)
    expected = arrow.get(expected)
    user.container.clock.utcnow = lambda: now
    date = user._get_next_date(hour, minute)
    assert date.tzinfo == dateutil.tz.tzutc()
    assert date == expected


@pytest.mark.parametrize('now, exp_start, exp_end', [
    ('2015-01-01T12:00:00', '2015-01-01T23:00:00', '2015-01-02T23:00:00'),
    ('2015-03-28T12:00:00', '2015-03-28T23:00:00', '2015-03-29T22:00:00'),
    ('2015-10-24T12:00:00', '2015-10-24T22:00:00', '2015-10-25T23:00:00'),
])
def test_user_get_dap_dates(user, now, exp_start, exp_end):
    now, exp_start, exp_end = [arrow.get(d) for d in [now, exp_start, exp_end]]
    ts = util.TimeSeries(exp_start, 900, [0 for i in range(96)])
    user._ts = ts
    s, e = user._get_dap_dates(user.container.clock)
    assert s.tzinfo == dateutil.tz.tzutc()
    assert s == exp_start
    assert e.tzinfo == dateutil.tz.tzutc()
    assert e == exp_end
