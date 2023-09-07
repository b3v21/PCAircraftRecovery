from collections import deque

# T0: F0 (depart A0, arrive A1) -> T0: F1 (depart A1, arrive A2)        P0
# T1: F2 (depart A0, arrive A1) -> T1: F3 (depart A1, arrive A2)        P1
# ...
# TN: F2N (depart A0, arrive A1) -> T4: F(2N+1) (depart A1, arrive A2)  PN

# Departures occuring every 0.5 hrs
# Arrivals occuring every

num_flights = 20
num_tails = 10
num_airports = 3
num_fare_classes = 2
num_delay_levels = 2

# Sets
T = range(num_tails)
F = range(num_flights)
P = [[i, i + 1] for i in range(0, num_flights, 2)]  # Set of itineraries
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
F_t = [list(F) for _ in T]

# set of tails compatible with flight F
T_f = [[t for t in T if f in F_t[t]] for f in F]

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
CF_f = [
    [
        fd
        for fd in F
        if AK_f[f] == departure_airport_of_f[fd] and sta[f] <= std[fd] and fd != f
    ]
    for f in F
]

# Subset of itineraries compatible with a reassignment from an original itinerary p.
# itinary p is compatible for a reassignment with itinary pd if they both share the
# same start and end destination

CO_p = [
    [
        P.index(pd)
        for pd in P
        if pd != []
        and departure_airport_of_f[pd[0]] == departure_airport_of_f[p[0]]
        and AK_f[pd[-1]] == AK_f[p[-1]]
    ]
    for p in P
    if p != []
]

# Data

# Cost of operating flight f with tail t
oc = [[1500 for _ in F] for _ in T]

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
CF_p = [(2 * p, 2 * p + 1) for p in range(len(P))]

# One if flight f is the last flight of itinerary p, and zero otherwise.
lf = [[(lambda last: 1 if last == f else 0)(p[1]) for p in P] for f in F]

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
x_hat_sub = [1, 1] + [0 for _ in range(num_flights - 2)]
x_hat = [x_hat_sub]

for _ in range(num_tails - 1):
    x_hat_sub = deque(x_hat_sub)
    x_hat_sub.rotate(2)
    x_hat_sub = list(x_hat_sub)
    x_hat += [x_hat_sub]
