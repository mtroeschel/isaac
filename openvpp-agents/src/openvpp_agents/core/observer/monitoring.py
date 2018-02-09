from asyncio import coroutine

import h5py

import numpy as np


class Monitoring:
    """Database for the storage of all agent and negotiation related
    data in the system.

    Stores values during negotation and writes them to hdf5-database at
    the end of a negotiation.
    """
    def __init__(self, dbfile):
        self._db = h5py.File(dbfile, mode='w')
        self._db.create_group('dap')
        self._topgroup = None
        self._agents = {}
        self._dap_data = []

    def setup(self, date):
        """Setup monitoring for a new run / negotiation.

        :param date: The begin date of the target_schedule for the
        negotitation.
        """
        self._agents = {}
        self._dap_data = []

        group_name = date.format('YYYYMMDD')
        db_group = self._db['/dap'].create_group(group_name)
        self._topgroup = db_group

    @coroutine
    def flush(self, target_schedule, weights, solution):
        """Writes *target_schedule*, *weights*, *solution* and collected
        negotation data to hdf5 database.

        :param target_schedule: A list of values indicating the
        electrical target for the negotiation.
        :param weights: A list of weights [0,1] indicating the
        optimization weight of the respective value in the
        *target_schedule*
        :param solution: Cluster schedule as
        :class:`openvpp_agents.planning.Candidate`
        """
        dap_data, self._dap_data = self._dap_data, []

        # Extract DAP data to be stored
        db_group = self._topgroup
        db_group.create_dataset('ts', data=target_schedule)
        # db_group.create_dataset('weights', data=weights)
        db_group.create_dataset('cs', data=solution.cs)

        self._store_data(db_group, dap_data)
        self._db.flush()

    def close(self):
        """Close the database."""
        self._db.close()

    def append(self, row):
        """Store *row* in Database, but do not flush - this is done in
        either :meth:stop or :meth:flush_collected_data
        """
        self._dap_data.append(row)

    def store_topology(self, connections):
        """Write topology between unit agents specified by *connections*
        to hdf5 database using dap/*date* as group.

        :param connections: The topology as list of tuples
        (agent_1, agent_2), to be understood as bidirectional
        connections.
        """
        assert self._topgroup
        # create encoded data set
        data = np.array([(a.encode(), b.encode())
                         for a, b in connections])
        # store data in group
        self._topgroup.create_dataset('topology', data=data)

    def _store_data(self, group, dap_data):
        dtype = np.dtype([
            ('t', 'float64'),
            ('agent', 'S100'),
            ('perf', 'float64'),
            ('complete', bool),
            ('msgs_out', 'uint64'),
            ('msgs_in', 'uint64'),
            ('msgs_sent', bool),
        ])
        dap_data = [(t, a.encode(), perf, complete, mo, mi, ms)
                    for t, a, perf, complete, mo, mi, ms in dap_data]
        dap_data = np.array(dap_data, dtype=dtype)
        group.create_dataset('dap_data', data=dap_data)
