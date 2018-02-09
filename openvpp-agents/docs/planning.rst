Open VPP COHDA implementation
=============================

The DAP agents implement a variant of COHDA [#hinsel]_ [#cohda]_ and DynaSCOPE
[#niesse]_.


Terms, variables and data structures
------------------------------------

Schedule
  Depending on the context a tuple *(start_date, interval_resolution,
  interval_values)*.  Each value in *interval_values* denotes the average
  active power feed-in / consumption during its interval in [W].  The interval
  resolution is a positive integer in [s].  The start date is an arrow[#arrow]_
  datetime object in UTC.

Operational Schedule *os / θ*
  A single unit's schedule.  Usually expressed just as the list of interval
  values: *os* = ndarray<float>(intervals)

Cluster Schedule *cs*
  A list of operational schedule from multiple or all agents within a VPP:
  *cs* => ndarray<float>(n, intervals)

Target Schedule *ts*
  The target schedule for a VHP:
  *ts* => ndarray<float>(intervals)

Counter vector *cnt*
  A vector that assigns every *os* of a *cs* as count / time stamp λ:
  *cnt* => list<int>(n)

Index map *idx*
  A dictionary mapping agent names to index values.  The index values point
  to the corresponding elements in a *cs* and *cnt* vector:
  *idx* => dict<str, int>(n)

System configuration *SystemConfig / sysconf*
  The system configuration of an agent contains the most recent operational
  schedules of the known agents that the agent is aware of.

  It is a tuple *(idx, cs, cnt)* where the index map *idx* can be used to get
  the index of an agent's *os* within *cs* as well as the corresponding time
  stamp *λ* within *cnt*.

  It is comparable (but not equal) to the Ω in the original COHDA.  Storing all
  *os* in a 2D NumPy array makes it easier and more efficient compute the
  *objective function* and to (de)serialize it when sending it to other agents.

Solution candidate *candidate*
  The best solution currently known to an agent.

  It is a tuple *(agent, idx, cs, perf)* where *agent* is the name of the agent
  that found this solution candidate.  The index map should be equal to the one
  of the corresponding *sysconf*.  The dimensions of the *cs* are the same as
  the ones from the *sysconf*, but the *cs* may contain different *os*.  *perf*
  is the performance computed by the *objective function*.  It is safe to
  cache, sice the *objective function* is global and all agents will compute
  the same performance given the same inputs.

Working memory *wm*
  The working memory (*κ* in the original COHDA) is the main data structure
  for the current planning process.

  It is a tuple *(start, res, intervals, ts, ps, sysconf, candidate)*.

  The first three values describe the *planning horizon*: *start* is an
  arrow[#arrow]_ datetime object in UTC.  *res* is the interval resolution in
  [s] and *intervals* an integer denoting number of intervals.

  *ts* is the *target schedule*, *ps* is a list of possible operational
  schedules *os* for a unit.  *sysconf* and *candidate* are the current system
  configuration and solution candidate.


Workflow
--------

The algorithm has three phases:

1. Initialization

2. Stepping

3. Stopping


Initialization
^^^^^^^^^^^^^^

The DAP controller initializes the whole process.  It first creates a new
overlay topology and sends a list of neighbors to each agent.  It’s not
necessary to compute a new topology for every DAP process, but this way it very
straight forward to integrate new agents or account for dead ones.

After that, the controller sends an *init* message to all agents.  This message
describes the planning horizon: *(start, res, intervals)*.

When an agent receives an *init* message, it first determines the list of
possible schedules *ps*.  This list won't change during this planning process.
It also creates an initial *sysconf* and *candidate* (which both only contain
one entry).  The agent has now all information for its working memory *wm*
and pushes a *step* message to its neighbors containing its *sysconf* and
*candidate*.


Stepping
^^^^^^^^

Agents process *step* message serially and in FIFO order until they receive a
*stop* message (see next section).

Each *step* is processed by performing the actions *perceive*, *decide*, and
*act*.

*perceive* merges the agent's *sysconf* and *candidate* with the ones received
from the other agent.  It returns the original *sysconf* or *candidate*
unchanged if the ones received from the other agent don't add anything new.

If either the *sysconf* or the *candidate* were updated, *decide* tries to
find a better candidate.  It first tries to find one that is locally better.
If such an *os* is found, it will call the objective function to check if it
is also globally better.  It returns a new *wm* instance with the updated
*sysconf* and *candidate*.

Finally, the agent sends a *step* message with its new *sysconf* and
*candiate* to all neighbors.


Stop
^^^^

When the DAP controller needs a result for the planning process, it sends a
*stop* message to all planning agents.

When an agent receives a *stop* message, it finishes processing the current
*step* message if necessary and resets its working memory and list of
neighbors.  It also returns its latest *sysconf* and *candidate* to the
controller.

The controller merges all candidates received to determine the final solution.
(If the algorithm terminated before the controller sent the *stop* message, all
candidates will be the same.)  Thus, a *stop* message can be sent at any time
and the algorithm will always produce a usable solution (although, not always
necessarily the best one).

There is currently now distributed termination detection (DTD) implemented that
allows to check if the algorithm has terminated without forcing it to stop.


Termination Detection
---------------------



.. [#hinsel] http://oops.uni-oldenburg.de/1960/1/hinsel14.pdf
.. [#cohda] Christian Hinrichs: *A distributed combinatorial optimisation
            heuristic for the scheduling of energy resources represented by
            self-interested agents*
.. [#niesse] Astrid Nieße: *Verteile kontinuierliche Einsatzplanung in
             Dynamischen Virtuellen Kraftwerken*
.. [#arrow] http://crsmithdev.com/arrow/
