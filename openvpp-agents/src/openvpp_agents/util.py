import functools
import itertools

import arrow
import aiomas.codecs
import click
import numpy as np


def get_container_kwargs(start_date):
    return {
        'clock': aiomas.ExternalClock(start_date, init_time=-1),
        'codec': aiomas.MsgPackBlosc,
        'extra_serializers': get_extra_serializers(),
    }


def validate_addr(ctx, param, value):
    try:
        host, port = value.rsplit(':', 1)
        return (host, int(port))
    except ValueError as e:
        raise click.BadParameter(e)


def validate_start_date(ctx, param, value):
    try:
        arrow.get(value)  # Check if the date can be parsed
    except arrow.parser.ParserError as e:
        raise click.BadParameter(e)
    return value


def get_np_serializer():
    """Return a tuple *(type, serialize(), deserialize())* for NumPy arrays
    for usage with an :class:`aiomas.codecs.MsgPack` codec.

    """
    return np.ndarray, _serialize_ndarray, _deserialize_ndarray


def _serialize_ndarray(obj):
    return {
        'type': obj.dtype.str,
        'shape': obj.shape,
        'data': obj.tostring(),
    }


def _deserialize_ndarray(obj):
    return np.fromstring(obj['data'],
                         dtype=np.dtype(obj['type'])).reshape(obj['shape'])


def get_extra_serializers():
    """Return the list of extra serializer functions used in openvpp."""
    from openvpp_agents.planning import Candidate, SystemConfig as SC
    return [
        get_np_serializer,
        SC.__serializer__,
        Candidate.__serializer__,
        TimeSeries.__serializer__,
    ]


def get_tomorrow(utc_now, tz, to_local=False):
    """Return an :class:`~arrow.arrow.Arrow` datetime for *tomorrow 00:00 local
    time*.

    The calculation is done relative to the UTC datetime *utc_start* converted
    to the timezone *tz*.

    By default, the result is converted back to UTC.  Set *to_local* to
    ``True`` to get a local date in timezone *tz*.

    Since *utc_start* is converted to *tz* first, this function takes care of
    local daylight saving time changes.

    **Example:** *utc_now* is *2015-01-15 13:37:00+00:00* in UTC.  If *tz* is
    *Europe/Berlin*, the date corresponds to *2015-01-15 14:37:00+01:00* in
    local time.  The returned date will be *2015-01-16 00:00:00+01:00* if
    *to_local* is ``True`` or *2015-01-15 23:00:00+00:00*  else.

    """
    today = utc_now.to(tz)
    today = today.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today.replace(days=1)
    out_tz = tz if to_local else 'utc'
    return tomorrow.to(out_tz)


def check_date_diff(date, base_date, res):
    """Assert that *date* >= *base_date* and that both dates are aligned with
    a resolution of *res* seconds.

    Raise a :exc:`ValueError` if this is not the case.  Else, return the number
    seconds between both dates.

    """
    diff = int((date - base_date).total_seconds())
    if diff < 0:
        raise ValueError('The date "%s" is not >= the base_date "%s"' %
                         (date, base_date))
    if diff % res != 0:
        raise ValueError('"%s" is at resolution of %d seconds not aligned '
                         'with "%s"' % (date, res, base_date))
    return diff


def get_intervals_between(target_date, base_date, res):
    """Return the number of intervals between the two dates *target_date* and
    *base_date* with a given interval resolution *res* in seconds.

    *target_date* must be greater or equal to *base_date*.  The difference
    between both dates must be a multiple of *res*.  Raise a :exc:`ValueError`
    if these constraints are violated.

    """
    diff = check_date_diff(target_date, base_date, res)
    return int(diff // res)


@aiomas.codecs.serializable
class TimeSeries:
    """TimeSeries is a wrapper for a NumPy array of *data* with a given
    *start* date and resolution *res* in seconds."""
    def __init__(self, start, res, data):
        self._start = arrow.get(start)
        self._res = int(res)
        self._data = np.array(data)

        self._idxfdate = functools.partial(get_intervals_between,
                                           base_date=self._start,
                                           res=self._res)

    def __len__(self):
        """Return the length of the time series, which is the same then the
        length of :attr:`data`."""
        return len(self._data)

    def __eq__(self, other):
        """Return ``True`` if *other* is equal to this time series and
        ``False`` if not.

        Two time series are considered equal if they have the same start date,
        resolution and data.

        """
        return (
            self._start == other.start and
            self._res == other.res and
            np.array_equal(self._data, other.data)
        )

    def __getitem__(self, i):
        """Extract values from :attr:`data`.

        Indices must be :class:`~arrow.arrow.Arrow` dates.  They'll be
        automatically converted to the correct indices for the underlying
        NumPy array.

        """
        if type(i) is slice:
            start = None if i.start is None else self._idxfdate(i.start)
            stop = self._idxfdate(i.stop)
            step = i.step
            i_new = slice(start, stop, step)
        elif type(i) is tuple:
            raise TypeError('indices must not be tuples')
        else:
            i_new = self._idxfdate(i)

        return self._data[i_new]

    def __setitem__(self, i, val):
        """Set data to :attr:`data`.

        Indices must be :class:`~arrow.arrow.Arrow` dates.  They'll be
        automatically converted to the correct indices for the underlying
        NumPy array.

        """
        if type(i) is slice:
            start = None if i.start is None else self._idxfdate(i.start)
            stop = self._idxfdate(i.stop)
            step = i.step
            i = slice(start, stop, step)
        elif type(i) is tuple:
            raise TypeError('indices must not be tuples')
        else:
            i = self._idxfdate(i)

        self._data[i] = val

    @property
    def start(self):
        """:class:`~arrow.arrow.Arrow` date denoting the start of the time
        series."""
        return self._start

    @property
    def end(self):
        """:class:`~arrow.arrow.Arrow` date denoting the end of the time
        series.  It is exclusive, meaning it is the first date for which no
        more data exist in the series."""
        return self._start.replace(seconds=self._res*len(self._data))

    @property
    def res(self):
        """The time series' resolution in seconds.

        Lower values mean a higher resolution.

        """
        return self._res

    @property
    def data(self):
        """The actual data of the time series."""
        return self._data

    @property
    def period(self):
        """The period in seconds that this time series covers.  This is the
        length of :attr:`data` times :attr:`res`."""
        return len(self._data) * self._res

    def copy(self):
        """Return a copy of the time series."""
        # __init__() creates copies of all attributes, so we don't need to do
        # it here.
        return self.__class__(self._start, self._res, self._data)

    def iter(self, start=None, res=None, n=None):
        """Return a generator for iterating over the time series.

        You can pass a date *start* if you don't want to start at the
        beginning.

        If you can also iterate in a non-native resolution if you provide
        a value for *res* (in seconds).  If *res* is lower then :attr:`res`
        (you request a higher resolution), you'll get the same values multiple
        times.  If you provide a higher value for *res* (lower actual
        resolution), some of the time series values will be skipped.

        """
        if start is None:
            start_idx = 0
        else:
            # self._idxfdate is a partial "get_intervals_between",
            # which also checks if the start date is fine.
            start_idx = self._idxfdate(start)
        if res is None:
            res = self._res

        rng = itertools.count() if n is None else range(n)

        try:
            for i in rng:
                i = i * res // self.res
                yield self._data[start_idx + i]
        except IndexError:
            raise StopIteration from None

    def extend(self, other):
        """Extend the :attr:`data` of this instance wit the data of *other*.

        If both series overlap, *other* overwrites this instance's data.

        """
        if self.res != other.res:
            raise ValueError('"other" must have a resolution of %d but has '
                             '%d' % (self.res, other.res))
        if not (self.start <= other.start <= self.end):
            raise ValueError('"other" must start between %s and %s but starts '
                             'at %s' % (self.start, self.end, other.start))

        idx = self._idxfdate(other.start)
        self._data = np.concatenate((self._data[:idx], other.data))

    def lstrip(self, strip_date, inclusive=False):
        """Strip all data from this time series < (or <= if *inclusive* is
        ``True``) *strip_date* and return a new time series containing the
        the stripped data.

        Raise a :exc:`ValueError` if *strip_date* or after :attr:`end`.

        """
        if strip_date > self.end:
            raise ValueError('"strip_date" must be <= %s but is %s' %
                             (self.end, strip_date))

        if strip_date < self.start:
            return None

        # Calculate the index where the split should happen
        diff = (strip_date - self.start).total_seconds()
        idx, remainder = divmod(diff, self.res)
        idx = int(idx)
        if inclusive and remainder == 0:
            idx += 1

        # Do the split
        stripped_data, self._data = self._data[:idx], self._data[idx:]
        stripped_ts = self.__class__(self._start, self._res, stripped_data)
        self._start = self.start.replace(seconds=len(stripped_data) * self.res)

        return stripped_ts

    def shift(self, new_start, fill=0):
        """Shift the time series to a new start date *new_start* and fill it up
        with *fill*.

        The shift is a left shift, so *new_start* must be >= the current start
        of time series.  Raise a :exc:`ValueError` if this is not the case.

        """
        diff = check_date_diff(new_start, self.start, self.res)
        if new_start == self.start:
            return

        self._start = new_start

        if new_start >= self.end:
            self._data = np.zeros(len(self._data))
            return

        shift = -(diff // self._res)

        self._data = np.roll(self._data, shift)
        self._data[shift:] = fill
