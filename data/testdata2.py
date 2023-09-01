import random
import numpy as np

# T0: F0 (depart A0, arrive A1) -> T10: F1 (depart A1, arrive A2)
# T1: F2 (depart A0, arrive A1) -> T11: F3 (depart A1, arrive A2)
# T2: F4 (depart A0, arrive A1) -> T12: F5 (depart A1, arrive A2)
# T3: F6 (depart A0, arrive A1) -> T13: F7 (depart A1, arrive A2)
# T4: F8 (depart A0, arrive A1) -> T14: F9 (depart A1, arrive A2)
# T5: F10 (depart A0, arrive A1) -> T15: F11 (depart A1, arrive A2)
# T6: F12 (depart A0, arrive A1) -> T16: F13 (depart A1, arrive A2)
# T7: F14 (depart A0, arrive A1) -> T17: F15 (depart A1, arrive A2)
# T8: F16 (depart A0, arrive A1) -> T18: F17 (depart A1, arrive A2)
# T9: F18 (depart A0, arrive A1) -> T19: F19 (depart A1, arrive A2)

# Departures occuring every 0.5 hrs
# Arrivals occuring every


random.seed(3)
num_flights = 20
num_tails = 20
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
DA = [(t, t + 1) for t in T]
AA = [(t - 0.5, t + 0.5) for t in range(1, num_tails + 1)]

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
F_t = []  # Assume this is right for now
for f in F:
    if f < 10:
        F_t += [list(range(0, 20, 2))]
    else:
        F_t += [list(range(1, 20, 2))]

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

# for f in F:
#     found = False
#     if not found:
#         for p in P:
#             if f not in p:
#                 continue
#             if f == p[0]:
#                 FD_k[random.randint(0, num_airports-1)].add(f)
#             else:
#                 FD_k[AK_f[p[p.index(f) - 1]]].add(f)
#             found = True

departure_airport_of_f = {}
for f in F:
    departure_airport_of_f[f] = -1

for f in F:
    for k in K:
        if f in FD_k[k]:
            departure_airport_of_f[f] = k

# Set of flights fd compatible with a connection from flight f
# fd is compatible if it is scheduled to depart from the arrival airport of flight f
CF_f = [
    [fd for fd in F if AK_f[f] == departure_airport_of_f[fd] and fd != f] for f in F
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
oc = [[1 for _ in range(num_flights)] for _ in range(num_tails)]

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
second = [[0, 1, 0]]

tb = []
for t in T:
    if t < 10:
        tb += first
    else:
        tb += second


print(len(tb))

# Capacity of arrival and departure slots
scA = [1 for _ in range(20)]
scD = [1 for _ in range(20)]
