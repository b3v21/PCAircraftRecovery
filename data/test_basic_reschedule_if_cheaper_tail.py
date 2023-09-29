"""
This will test that a flight is flown is reassigned to a cheaper tail
Reassigns F0: (depart A0, arrive A1) from T0 to T1 due to large cost to fly T0
"""


num_flights = 1
num_tails = 2
num_airports = 2
num_fare_classes = 2
num_delay_levels = 2

# Sets
T = range(num_tails)
F = range(num_flights)
P = [[0]]  # Set of itineraries

K = range(num_airports)
Y = range(num_fare_classes)
Z = range(num_delay_levels)

# Scheduled arrival (departure) time for flight f in F
std = [f + 0.5 for f in F]
sta = [f + 1 for f in F]

# Arrival and Depature slots
DA = [(f, f + 1) for f in F]
AA = [(f - 0.5, f + 0.5) for f in range(1, num_flights + 1)]

# Set of arrival and departure slots compatible with flight f
AAF = [
    [i for i, slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]]
    for f in F
]
DAF = [
    [i for i, slot in enumerate(DA) if std[f] <= slot[1] and std[f] >= slot[0]]
    for f in F
]

# Set of flights compatible with arrive/departure slot asl/dsl
FAA = [[f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA]
FDA = [[f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA]

# set of flights compatible with tail T
# (currently every flight is compatible with every tail)
F_t = [[0], [0]]

# set of tails compatible with flight F
T_f = [[0, 1]]

# Set of flights f which arrive to airport k
flights = [f for f in F]
FA_k = {}
for k in K:
    FA_k[k] = set()

AK_f = {}  # Airport that flight f arrives at (this isnt actually data in the paper)

for f in F:
    if f % 2 == 0:
        FA_k[1].add(f)
        AK_f[f] = 1
    else:
        FA_k[2].add(f)
        AK_f[f] = 2

# Set of flights f which depart from airport k
FD_k = {}
for k in K:
    FD_k[k] = set()

for f in F:
    if f % 2 == 0:
        FD_k[0].add(f)
    else:
        FD_k[1].add(f)

departure_airport_of_f = {}
for f in F:
    departure_airport_of_f[f] = -1

for f in F:
    for k in K:
        if f in FD_k[k]:
            departure_airport_of_f[f] = k

# Set of flights fd compatible with a connection from flight f
# fd is compatible if it is scheduled to depart from the arrival airport of flight f
# and the scheduled arrival of f is before the scheduled departure of fd
CF_f = [[]]

# Subset of itineraries compatible with a reassignment from an original itinerary p.
# itinary p is compatible for a reassignment with itinary pd if they both share the
# same start and end destination

CO_p = [[0]]

# Data

# Cost of operating flight f with tail t (oc[t][f])
oc = [[1000000000], [100]]

# Delay cost per minute of arrival delay of flight f
dc = [100 for _ in F]

# Number of passengers in fare class v that are originally scheduled to
# take itinerary p
n = [[50 for _ in P] for _ in Y]

# Seating capacity of tail t in T
q = [100 for _ in T]

# Reaccommodation Cost for a passenger reassigned from p to pd.
rc = [[(lambda p, pd: 0 if p == pd else 0.5)(p, pd) for p in P] for pd in P]

# Phantom rate for passenger in fare class v reassigned from p to pd with delay level
# zeta
theta = [[[[0 for _ in Y] for _ in P] for _ in P] for _ in Z]

# Starting location of planes (binary) ((for t) for k)
first = [[1, 0, 0]]
second = [[1, 0, 0]]

tb = []
for t in T:
    if t < num_tails / 2:
        tb += first
    else:
        tb += second


print(len(tb))

# Capacity of arrival and departure slots
scA = [1 for _ in F]
scD = [1 for _ in F]

# Scheduled buffer time for each flight (set to 0 for now)
sb = [0 for _ in F]

# minimum turn time between flight f and fd with tail t
mtt = [[[0 for _ in T] for _ in F] for _ in F]

# minimum connection time between flight f and fd in itinerary p
mct = [[[0 for _ in P] for _ in F] for _ in F]

# Planned connection time between flights f and fd. It equals scheduled departure time of
# flight fd minus the scheduled arrival time of flight f.
ct = [[max(0, std[fd] - sta[f]) for fd in F] for f in F]

# set of ordered flight pairs of consecutive flights in itinary p.
CF_p = [()]

# One if flight f is the last flight of itinerary p, and zero otherwise.
lf = [[1]]

# Upper bound on the delay, expressed in minutes, corresponding to delay level ζ.
small_theta = [1000 for _ in Z]

# Extra fuel cost for delay absorption (through cruise speed increases) per minute for
# flight f.
fc = [100 for _ in F]

# Sum of the cost of the loss of goodwill and the compensation cost (if any) for a
# passenger who was scheduled to take itinerary p and is reassigned to itinerary p’, if
# the passenger’s destination arrival delay via itinerary p′ compared with the planned
# arrival time of itinerary p corresponds to delay level ζ
pc = [[[0 for _ in Z] for _ in P] for _ in P]

# Per-flight schedule change penalty for not operating the flight using the originally
# planned tail.
kappa = 100

# One if flight f was originally scheduled to be operated by tail t, and zero otherwise.
x_hat = [[1], [0]]
