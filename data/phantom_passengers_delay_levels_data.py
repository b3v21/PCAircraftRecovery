from collections import deque

"""
This is a test for checking passenger reassignment between itineraries

Original:
    Itineraries:        P0: (F0, F1, F3)
                        P1: (F0, F2, F4)
                        
    Flights:
                        F0: depart A0 @1 min, arrive A1 @61 min
                        F1: depart A1 @86 min, arrive A3 @116 min
                        F2: depart A1 @106 min, arrive A2 @126 min
                        F3: depart A3 @136 min, arrive A4 @196 min
                        F4: depart A2 @146 min, arrive A4 @206 min
                        
    Tails:
                        T0: F0
                        T1: F1
                        T2: F2
                        T3: F3
                        T4: F4
                        
    Airports:
                        A0
                        A1
                        A2
                        A3
                        A4
                        
                        
Goal Reassignment: None 

Air transport compensation (EU Regulation)
    -> Australian Regulation for flight delays & cancellation are airline specific,
    with specific costs being difficult to determine (e.g. meal costs, rebooking costs etc.)
   
    1. Flights <= 1500 km
        Denied Boarding
        a) EUR €250 if rerouted and arriving later by two hours or more
        b) EUR €125 if rerouted and arriving less than two hours late
    
    2. Flights 1500-3500 km, EU flights >= 1500km
        Denied Boarding
        a) EUR €400 if rerouted and arriving later by three hours or more
        b) EUR €200 if rerouted and arriving less than three hours late
    
    3. Flights >= 3500 km (Excluding EU flights >= 1500km)
        Denied Boarding
        a) EUR €600 if rerouted and arriving later by four hours or more
        b) EUR €300 if rerouted and arriving less than four hours late

Delay Levels
    1,2,3   -> Inconvenience costs associated with loss of passenger goodwill
    4,5     -> Inconvenience cost & compensation cost
    
    Level 1 -> 0-59 minutes (<1 hour)
    Level 2 -> 60-119 minutes (>=1 hour & <2 hours)
    Level 3 -> 120-179 minutes (>=2 hour & <3 hours)
    level 4 -> 180-239 minutes (>=3 hour & <4 hours)
    level 5 -> >=240 minutes
    
"""


num_flights = 5
num_tails = 5
num_airports = 5
num_fare_classes = 15
num_delay_levels = 5


########
# Sets #
########
T = range(num_tails)  # Set of tails
F = range(num_flights)  # Set of flights
P = [[0, 1, 3], [0, 2, 4]]  # Set of itineraries
K = range(num_airports)  # Set of airports
Y = range(num_fare_classes)  # Set of fare classes
Z = range(num_delay_levels)  # Set of delay levels

# Scheduled arrival (departure) time for flight f in F
std = {0: 1, 1: 86, 2: 106, 3: 136, 4: 146}
sta = {0: 61, 1: 116, 2: 126, 3: 196, 4: 206}

# Arrival and Depature slots
DA = [(0, 15), (81, 96), (101, 116), (131, 146), (141, 156)]
AA = [(56, 71), (111, 126), (121, 136), (191, 206), (201, 216)]

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
FA_k = {0: [], 1: [0], 2: [2], 3: [1], 4: [4]}

# Airport that flight f arrives at (this isnt actually data in the paper)
AK_f = {0: 1, 1: 3, 2: 2, 3: 4, 4: 4}

# Set of flights f which depart from airport k
FD_k = {0: [0], 1: [1, 2], 2: [4], 3: [3], 4: []}

# Airport that flight f departs from (this isn't actually data in the paper)
DK_f = {0: 0, 1: 1, 2: 1, 3: 3, 4: 2}

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
CF_f = {
    f: [
        fd
        for fd in F
        if AK_f[f] == departure_airport_of_f[fd] and sta[f] <= std[fd] and fd != f
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
        and std[pd[0]] >= std[p[0]]
        and sta[pd[-1]] >= sta[p[-1]]
    ]
    for p in P
    if p != []
}


########
# Data #
########

# Cost of operating flight f with tail t
oc = {(t, f): 500 for f in F for t in T}

# Delay cost per minute of arrival delay of flight f
dc = {f: 100 for f in F}

# Number of passengers in fare class v that are originally scheduled to
# take itinerary p
n = {(v, P.index(p)): 25 for v in Y for p in P}

# Seating capacity of tail t in T
q = {t: 250 for t in T}

# Reaccommodation Cost for a passenger reassigned from p to pd.
rc = {
    (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 0.5)(p, pd)
    for p in P
    for pd in P
}

alpha_regr = {0: 0, 1: 0, 2: 0, 3: 0.290, 4: 0.473}

beta_regr = {0: 0, 1: 0, 2: 0, 3: -0.016, 4: -0.028}

# Phantom rate for passenger in fare class v reassigned from p to pd with delay level
# zeta
theta = {
    (v, P.index(p_to), P.index(p_from), z): alpha_regr[z] + beta_regr[z] * v
    for z in Z
    for p_from in P
    for p_to in P
    for v in Y
}
# print(theta)

# Starting location of planes (binary) ((for t) for k)
# t0 = [1, 0, 0, 0, 0]
# t1 = [0, 1, 0, 0, 0]
# t2 = [0, 1, 0, 0, 0]
# t3 = [0, 0, 0, 1, 0]
# t4 = [0, 0, 1, 0, 0]

# tb = {0:t0, 1:t1, 2:t2, 3:t3, 4:t4}

tb = {(t, k): 0 for t in T for k in K}
tb[(0, 0)] = 1
tb[(1, 1)] = 1
tb[(2, 1)] = 1
tb[(3, 3)] = 1
tb[(4, 2)] = 1
# print(tb)

# Capacity of arrival and departure slots
scA = {asl: 10 for asl in AA}
scD = {asl: 10 for asl in DA}

# Scheduled buffer time for each flight (set to 0 for now)
sb = {f: 0 for f in F}

# minimum turn time between flight f and fd with tail t
mtt = {(t, f, fd): 0 for t in T for fd in F for f in F}

# minimum connection time between flight f and fd in itinerary p
mct = {(P.index(p), f, fd): 0 for p in P for fd in F for f in F}

# Planned connection time between flights f and fd. It equals scheduled departure time of
# flight fd minus the scheduled arrival time of flight f.
ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}

# set of ordered flight pairs of consecutive flights in itinary p.
CF_p = {0: [(0, 1), (1, 3)], 1: [(0, 2), (2, 4)]}

# One if flight f is the last flight of itinerary p, and zero otherwise.
lf = {(P.index(p), f): 0 for p in P for f in F}
lf[0, 3] = 1
lf[1, 4] = 1

# Upper bound on the delay, expressed in minutes, corresponding to delay level ζ.
small_theta = {0: 59, 1: 119, 2: 179, 3: 239, 4: 1000}

# Extra fuel cost for delay absorption (through cruise speed increases) per minute for
# flight f.
fc = {f: 100 for f in F}

# Sum of the cost of the loss of goodwill and the compensation cost (if any) for a
# passenger who was scheduled to take itinerary p and is reassigned to itinerary p’, if
# the passenger’s destination arrival delay via itinerary p′ compared with the planned
# arrival time of itinerary p corresponds to delay level ζ
pc = {(z, P.index(p), P.index(pd)): 100 for z in Z for p in P for pd in P}

# Per-flight schedule change penalty for not operating the flight using the originally
# planned tail.

# Kappa = 2000 as in paper
kappa = 2000

# One if flight f was originally scheduled to be operated by tail t, and zero otherwise.
x_hat = {(f,t): 0 for t in T for f in F}
x_hat[(0,0)]=1
x_hat[(1,1)]=1
x_hat[(2,2)]=1
x_hat[(3,3)]=1
x_hat[(4,4)]=1

import pdb; pdb.set_trace() #Debugging tool
