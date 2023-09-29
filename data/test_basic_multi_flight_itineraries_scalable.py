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
std = {f: f + 0.5 for f in F}
sta = {f: f + 1 for f in F}

# Arrival and Depature slots
DA = [(f, f + 1) for f in F]
AA = [(f - 0.5, f + 0.5) for f in range(1, num_flights + 1)]

# Set of arrival and departure slots compatible with flight f
AAF = {
    f: [i for i, slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]]
    for f in F
}

DAF = {
    f: [i for i, slot in enumerate(DA) if std[f] <= slot[1] and std[f] >= slot[0]]
    for f in F
}

# Set of flights compatible with arrive/departure slot asl/dsl
FAA = {asl: [f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA}
FDA = {dsl: [f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA}

# set of flights compatible with tail T
# (currently every flight is compatible with every tail)
F_t = {t: list(F) for t in T}

# set of tails compatible with flight F
T_f = {f: [t for t in T if f in F_t[t]] for f in F}

# Set of flights f which arrive to airport k
flights = [f for f in F]
FA_k = {}
for k in K:
    FA_k[k] = []

AK_f = {}  # Airport that flight f arrives at (this isnt actually data in the paper)

for f in F:
    if f % 2 == 0:
        FA_k[1].append(f)
        AK_f[f] = 1
    else:
        FA_k[2].append(f)
        AK_f[f] = 2

# Set of flights f which depart from airport k
FD_k = {}
for k in K:
    FD_k[k] = []

for f in F:
    if f % 2 == 0:
        FD_k[0].append(f)
    else:
        FD_k[1].append(f)

DK_f = {}
for f in F:
    DK_f[f] = -1

for f in F:
    for k in K:
        if f in FD_k[k]:
            DK_f[f] = k

# Set of flights fd compatible with a connection from flight f
# fd is compatible if it is scheduled to depart from the arrival airport of flight f
# and the scheduled arrival of f is before the scheduled departure of fd
CF_f = {
    f: [
        fd
        for fd in F
        if AK_f[f] == DK_f[fd] and sta[f] <= std[fd] and fd != f
    ]
    for f in F
}

# Subset of itineraries compatible with a reassignment from an original itinerary p.
# itinary p is compatible for a reassignment with itinary pd if they both share the
# same start and end destination

CO_p = {
    P.index(p): [
        P.index(pd)
        for pd in P
        if pd != []
        and DK_f[pd[0]] == DK_f[p[0]]
        and AK_f[pd[-1]] == AK_f[p[-1]]
    ]
    for p in P
    if p != []
}

# Data

# Cost of operating flight f with tail t
oc = {(t, f): 1500 for f in F for t in T}

# Delay cost per minute of arrival delay of flight f
dc = {f: 100 for f in F}

# Number of passengers in fare class v that are originally scheduled to
# take itinerary p
n = {(v, P.index(p)): 50 for p in P for v in Y}

# Seating capacity of tail t in T
q = {t: 100 for t in T}

# Reaccommodation Cost for a passenger reassigned from p to pd.
rc = {
    (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 0.5)(p, pd)
    for p in P
    for pd in P
}

# Phantom rate for passenger in fare class v reassigned from p to pd with delay level
# zeta
theta = {
    (v, P.index(p), P.index(pd), z): 0 for v in Y for p in P for pd in P for z in Z
}

tb = {(t, k): 1 if k == 0 else 0 for t in T for k in K}

# Capacity of arrival and departure slots
scA = {asl: 1 for asl in AA}
scD = {dsl: 1 for dsl in DA}

# Scheduled buffer time for each flight (set to 0 for now)
sb = {f: 0 for f in F}

# minimum turn time between flight f and fd with tail t
mtt = {(t, f, fd): 0 for t in T for f in F for fd in F}

# minimum connection time between flight f and fd in itinerary p
mct = {(P.index(p), f, fd): 0 for p in P for f in F for fd in F}

# Planned connection time between flights f and fd. It equals scheduled departure time of
# flight fd minus the scheduled arrival time of flight f.
ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}

# set of ordered flight pairs of consecutive flights in itinary p.
CF_p = {P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])] for p in P}

# One if flight f is the last flight of itinerary p, and zero otherwise.
lf = {(P.index(p), f): 1 if p[-1] == f else 0 for p in P for f in F}

# Upper bound on the delay, expressed in minutes, corresponding to delay level ζ.
small_theta = {z: 1000 for z in Z}

# Extra fuel cost for delay absorption (through cruise speed increases) per minute for
# flight f.
fc = {f: 100 for f in F}

# Sum of the cost of the loss of goodwill and the compensation cost (if any) for a
# passenger who was scheduled to take itinerary p and is reassigned to itinerary p’, if
# the passenger’s destination arrival delay via itinerary p′ compared with the planned
# arrival time of itinerary p corresponds to delay level ζ
pc = {(z, P.index(p), P.index(pd)): 0 for z in Z for p in P for pd in P}

# Per-flight schedule change penalty for not operating the flight using the originally
# planned tail.
kappa = 100

# One if flight f was originally scheduled to be operated by tail t, and zero otherwise.
x_hat = {(f, t): 1 if t == f or t + 1 == f else 0 for f in F for t in T}
