from collections import deque

"""
This is a test for checking passenger reassignment between itineraries

Original:               (T0) F0 (depart A0, arrive A1)   ITINERARY: 0
Goal Reassignment:      (T1) F1 (depart A0, arrive A1)   ITINERARY: 1
"""

num_flights = 2
num_tails = 2
num_airports = 2
num_fare_classes = 2
num_delay_levels = 2

# Sets
T = range(num_tails)
F = range(num_flights)
P = [[0],[1]] # Set of itineraries
K = range(num_airports)
Y = range(num_fare_classes)
Z = range(num_delay_levels)

# Scheduled arrival (departure) time for flight f in F
std = [1, 6]
sta = [3, 8]

# Arrival and Depature slots
DA = [(0,2),(5,7)]
AA = [(2,4),(7,9)] 

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
FA_k = {0:[],1:[0,1]}

# Airport that flight f arrives at (this isnt actually data in the paper)
AK_f = {0: 1, 1 : 1} 

# Set of flights f which depart from airport k
FD_k = {0:[0,1], 1:[]}

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
CF_f = [[] for fd in F]

# Subset of itineraries compatible with a reassignment from an original itinerary p.
# itinary p is compatible for a reassignment with itinary pd if they both share the
# same start and end destination

CO_p = [[0,1],[1]]

# Data

# Cost of operating flight f with tail t
oc = [[500 for _ in F] for _ in T]

# Delay cost per minute of arrival delay of flight f
dc = [1 for _ in F]

# Number of passengers in fare class v that are originally scheduled to
# take itinerary p
n = [[25, 25] for _ in Y]

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
CF_p = [(),()]

# One if flight f is the last flight of itinerary p, and zero otherwise.
lf = [[0,0],[1,1]]

# Upper bound on the delay, expressed in minutes, corresponding to delay level ζ.
small_theta = [1000 for _ in Z]

# Extra fuel cost for delay absorption (through cruise speed increases) per minute for
# flight f.
fc = [1 for _ in F]

# Sum of the cost of the loss of goodwill and the compensation cost (if any) for a
# passenger who was scheduled to take itinerary p and is reassigned to itinerary p’, if
# the passenger’s destination arrival delay via itinerary p′ compared with the planned
# arrival time of itinerary p corresponds to delay level ζ
pc = [[[100 for _ in Z] for _ in P] for _ in P]

# Per-flight schedule change penalty for not operating the flight using the originally
# planned tail.
kappa = 100

# One if flight f was originally scheduled to be operated by tail t, and zero otherwise.
x_hat = [[1,0],[0,1]]

