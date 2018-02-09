import io

import aiomas.clocks
import aiomas.codecs
import arrow
import numpy as np
import pytest

from openvpp_agents import planning
from openvpp_agents import util


TS = util.TimeSeries


@pytest.mark.parametrize('Codec', [
    aiomas.codecs.MsgPack,
    aiomas.codecs.MsgPackBlosc,
])
def test_np_serialize(Codec):
    x = np.arange(10, dtype='f8')
    codec = Codec()
    codec.add_serializer(*util.get_np_serializer())

    y = codec.decode(codec.encode(x))
    assert np.array_equal(y, x)


@pytest.mark.parametrize('Codec', [
    aiomas.codecs.MsgPack,
    aiomas.codecs.MsgPackBlosc,
])
def test_get_extra_codecs(Codec):
    codec = Codec()
    codec.add_serializer(*aiomas.util.arrow_serializer())
    for s in util.get_extra_serializers():
        codec.add_serializer(*s())

    sc = planning.SystemConfig({'1': 0}, np.arange(1), [1], [2])
    ret = codec.decode(codec.encode(sc))
    assert ret == sc

    cand = planning.Candidate('foo', {'1': 0}, np.arange(1), [1], 2)
    ret = codec.decode(codec.encode(cand))
    assert ret == cand

    ts = TS(arrow.get(), 15, np.arange(4))
    ret = codec.decode(codec.encode(ts))
    assert ret == ts


@pytest.mark.parametrize(['utc_start', 'to_local', 'expected'], [
    # 2015-03-28 00:00 Europe/Berlin (CET)
    ('2015-03-27 23:00', True,  '2015-03-29 00:00'),
    ('2015-03-27 23:00', False, '2015-03-28 23:00'),
    # 2015-03-29 01:00 Europe/Berlin (CET)
    ('2015-03-28 00:00', False, '2015-03-28 23:00'),
    # 2015-03-29 02:30 Europe/Berlin (CET)
    ('2015-03-28 01:30', False, '2015-03-28 23:00'),
    # 2015-03-29 00:00 Europe/Berlin (CET)
    ('2015-03-28 23:00', False, '2015-03-29 22:00'),  # local day has 23h
    # 2015-03-29 01:00 Europe/Berlin (CET)
    ('2015-03-29 00:00', False, '2015-03-29 22:00'),  # local day has 23h
    # 2015-03-29 02:30 Europe/Berlin (CET)
    ('2015-03-29 01:30', False, '2015-03-29 22:00'),  # local day has 23h
    # 2015-03-30 00:00 Europe/Berlin (CEST)
    ('2015-03-29 22:00', False, '2015-03-30 22:00'),
    # 2015-10-24 00:00 Europe/Berlin (CEST)
    ('2015-10-23 22:00', False, '2015-10-24 22:00'),
    # 2015-10-25 00:00 Europe/Berlin (CEST)
    ('2015-10-24 22:00', False, '2015-10-25 23:00'),  # local day has 25h
    # 2015-10-26 00:00 Europe/Berlin (CET)
    ('2015-10-25 23:00', False, '2015-10-26 23:00'),
])
def test_get_tomorrow(utc_start, to_local, expected):
    utc_start = '%s:00+00:00' % utc_start
    utc_start = arrow.get(utc_start.replace(' ', 'T'))
    ret = util.get_tomorrow(utc_start, 'Europe/Berlin', to_local)
    assert ret.format('YYYY-MM-DD HH:mm') == expected


@pytest.mark.parametrize('date, expected', [
    ('2015-01-01T00:00:00', 0),
    ('2015-01-01T00:00:30', 30),
])
def test_check_date_diff(date, expected):
    date = arrow.get(date)
    base_date = arrow.get('2015-01-01')
    diff = util.check_date_diff(date, base_date, 30)
    assert diff == expected


def test_check_date_diff_date_too_old():
    """An error should be raised if *date* is before *base_date*."""
    pytest.raises(ValueError, util.check_date_diff,
                  arrow.get('2015-01-01'), arrow.get('2015-01-02'), 900)


def test_check_date_diff_alignment_error():
    """An error should be raised if *date* does not align with *base_date*
    and *res*."""
    base_date = arrow.get('2015-01-01')
    date = base_date.replace(minutes=10)
    pytest.raises(ValueError, util.check_date_diff,
                  date, base_date, 900)


@pytest.mark.parametrize(['base_date', 'target_date', 'res', 'expected'], [
    ('2015-03-27 23:00', '2015-03-28 23:00', 15, 96),
    ('2015-03-28 23:00', '2015-03-29 22:00', 15, 92),
    ('2015-03-28 00:00', '2015-03-28 02:00', 1, 120),
])
def test_get_intervals_between(base_date, target_date, res, expected):
    base_date = arrow.get(base_date, 'YYYY-MM-DD HH:mm')
    target_date = arrow.get(target_date, 'YYYY-MM-DD HH:mm')
    res *= 60  # Convert to seconds
    result = util.get_intervals_between(target_date, base_date, res)
    assert result == expected
    assert type(result) is int


def test_ts():
    start = arrow.get('2015-01-01')
    res = 900
    data = [0, 1]
    ts = TS(start, res, data)
    assert ts.start is not start
    assert ts.start == arrow.get('2015-01-01')
    assert ts.end == arrow.get('2015-01-01T00:30:00')
    assert ts.res == 900
    assert ts.data is not data
    assert type(ts.data) is np.ndarray
    assert list(ts.data) == data
    assert len(ts) == 2
    assert ts.period == 30 * 60  # 30 minutes


def test_get_single_item():
    start = arrow.get('2015-01-01')
    ts = TS(start, 60, range(4))

    assert ts[start] == 0
    with pytest.raises(IndexError):
        ts[start.replace(minutes=5)]
    with pytest.raises(TypeError):
        ts[start, start]  # Multi-dimensional access is not implemented


def test_set_single_item():
    start = arrow.get('2015-01-01')
    ts = TS(start, 60, range(4))
    # Set single item
    ts[start.replace(minutes=2)] = 23
    assert list(ts.data) == [0, 1, 23, 3]
    with pytest.raises(IndexError):
        ts[start.replace(minutes=5)] = 23
    with pytest.raises(TypeError):
        ts[start, start] = 23  # Multi-dimensional access is not implemented


def test_get_slice():
    start = arrow.get('2015-01-01')
    ts = TS(start, 60, range(4))

    assert list(ts[start:start.replace(minutes=2)]) == [0, 1]
    assert list(ts[start.replace(minutes=5):start.replace(minutes=7)]) == []
    with pytest.raises(TypeError):
        ts[start:start, start:start]


def test_set_slice():
    start = arrow.get('2015-01-01')
    ts = TS(start, 60, range(4))
    # Set single item
    ts[start.replace(minutes=2):start.replace(minutes=4)] = [23, 42]
    assert list(ts.data) == [0, 1, 23, 42]
    ts[start.replace(minutes=5):start.replace(minutes=6)] = [8]
    assert list(ts.data) == [0, 1, 23, 42]  # Unchanged!
    with pytest.raises(TypeError):
        ts[start:start, start:start] = [23]


def test_ts_copy():
    ts1 = TS(arrow.get(), 1, [0, 1])
    ts2 = ts1.copy()
    assert ts2 is not ts1
    assert ts2 == ts1
    assert ts2.start is not ts1.start
    assert ts2.data is not ts1.data


@pytest.mark.parametrize('res, expected', [
    (60, list(range(30))),
    (30, [i // 2 for i in range(60)]),
    (900, [0, 15]),
])
def test_ts_iter_from_start(res, expected):
    ts = TS(arrow.get('2015-01-01'), 60, list(range(30)))
    ret = list(ts.iter(res=res))
    assert ret == expected


@pytest.mark.parametrize('res, expected', [
    (60, list(range(15, 30))),
    (30, [i // 2 for i in range(30, 60)]),
    (900, [15]),
])
def test_ts_iter_from_mid(res, expected):
    ts = TS(arrow.get('2015-01-01'), 60, list(range(30)))
    ret = list(ts.iter(start=ts.start.replace(minutes=15), res=res))
    assert ret == expected


def test_ts_iter_stop_iteration():
    ts = TS(arrow.get('2015-01-01'), 60, [0, 1])
    # Get too much
    ret = list(ts.iter(n=10))
    assert ret == list(ts.data)

    # Start too late
    ret = list(ts.iter(start=ts.start.replace(days=1), n=2))
    assert ret == []


@pytest.mark.parametrize('other_start, expected', [
    ('2015-01-01T00:00:00', [2, 3]),
    ('2015-01-01T00:15:00', [0, 2, 3]),
    ('2015-01-01T00:30:00', [0, 1, 2, 3]),
])
def test_ts_extend(other_start, expected):
    start = arrow.get('2015-01-01')
    res = 900
    ts = TS(start, res, np.array([0, 1]))
    ts.extend(TS(arrow.get(other_start), res, [2, 3]))
    assert ts == TS(start, res, expected)
    assert ts.end == start.replace(seconds=len(expected)*res)
    assert len(ts) == len(expected)


def test_ts_extend_wrong_res():
    ts = TS(arrow.get('2015-01-01'), 60, [0, 1, 2, 3])
    other = TS(arrow.get('2015-01-01T00:02'), 30, [0, 1, 2, 3])
    pytest.raises(ValueError, ts.extend, other)


def test_ts_extend_wrong_start():
    ts = TS(arrow.get('2015-01-01'), 60, [0, 1, 2, 3])
    other = TS(arrow.get('2015-01-01T00:05'), 60, [0, 1, 2, 3])
    pytest.raises(ValueError, ts.extend, other)
    other = TS(other.start.replace(minutes=-6), 60, [0, 1, 2, 3])
    pytest.raises(ValueError, ts.extend, other)


@pytest.mark.parametrize('lstrip, stripped_data, new_data, new_start', [
    ('2015-01-01T00:00:00', [], [0, 1, 2, 3], '2015-01-01T00:00:00'),  # start
    ('2015-01-01T00:30:00', [0, 1],   [2, 3], '2015-01-01T00:30:00'),  # exact
    ('2015-01-01T00:50:00', [0, 1, 2],   [3], '2015-01-01T00:45:00'),  # betw.
    ('2015-01-01T01:00:00', [0, 1, 2, 3], [], '2015-01-01T01:00:00'),  # end
])
def test_ts_lstrip(lstrip, stripped_data, new_data, new_start):
    start = arrow.get('2015-01-01')
    ts = TS(start, 900, [0, 1, 2, 3])
    stripped = ts.lstrip(arrow.get(lstrip))
    assert stripped is not ts
    assert stripped.start == start
    assert list(stripped.data) == stripped_data
    assert ts.start == arrow.get(new_start)
    assert list(ts.data) == new_data


def test_ts_lstrip_inclusive():
    """If the lstrip-date falls on a discrete value and *lstrip_to_right* was set
    to ``False``, that value should be at the end of the left list and not at
    the beginning of the right one."""
    # Thats the normal way
    ts = TS(arrow.get('2015-01-01'), 60, [0, 1, 2])
    stripped = ts.lstrip(ts.start.replace(minutes=1))
    assert list(stripped.data) == [0]
    assert list(ts.data) == [1, 2]

    # Now the 1 should be in the left list
    ts = TS(arrow.get('2015-01-01'), 60, [0, 1, 2])
    stripped = ts.lstrip(ts.start.replace(minutes=1), inclusive=True)
    assert list(stripped.data) == [0, 1]
    assert list(ts.data) == [2]


def test_ts_lstrip_out_of_bounds():
    ts = TS(arrow.get('2015-01-01'), 60, [0, 1, 2, 3])
    pytest.raises(ValueError, ts.lstrip, ts.start.replace(minutes=5))
    ret = ts.lstrip(ts.start.replace(minutes=-1))
    assert ret is None
    assert len(ts) == 4


@pytest.mark.parametrize('shift, expected', [
    ('2015-01-01T00:00:00', [0, 1, 2, 3]),
    ('2015-01-01T00:01:00', [1, 2, 3, 0]),
    ('2015-01-01T00:02:00', [2, 3, 0, 0]),
    ('2015-01-01T00:03:00', [3, 0, 0, 0]),
    ('2015-01-01T00:04:00', [0, 0, 0, 0]),
    ('2015-01-01T00:05:00', [0, 0, 0, 0]),
])
def test_ts_shift(shift, expected):
    shift = arrow.get(shift)
    ts = TS(arrow.get('2015-01-01'), 60, [0, 1, 2, 3])
    ts.shift(shift)
    assert ts.start == shift
    assert list(ts.data) == expected


def test_ts_shift_fill_value():
    ts = TS(arrow.get('2015-01-01'), 60, [0, 1, 2, 3])
    ts.shift(arrow.get('2015-01-01T00:02:00'), 23)
    assert list(ts.data) == [2, 3, 23, 23]
