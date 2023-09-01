import random

random.seed(3)
num_flights = 80
num_tails = 20
num_airports = 4
num_fare_classes = 2
num_delay_levels = 2

# Sets
T = range(num_tails)
F = range(num_flights)
P = [[i,i+1] for i in range(0,79,2)]  # Set of itineraries
K = range(num_airports)
Y = range(num_fare_classes)
Z = range(num_delay_levels)

# Scheduled arrival (departure) time for flight f in F
std = [f + 0.5 for f in F]
sta = [f + 1.5 for f in F]

# Arrival and Depature slots
DA = [(t, t + 1) for t in T]
AA = [(t, t + 1) for t in range(1, 21)]

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
F_t = [list(range(0 + 4 * t, 4 + 4 * t)) for t in T]  # Assume this is right for now

# set of tails compatible with flight F
T_f = [[t for t in T if f in F_t[t]] for f in F]

# Set of flights f which arrive to airport k
flights = [f for f in F]
FA_k = {}
AK_f = {}  # Airport that flight f arrives at (this isnt actually data in the paper)

while flights:
    for k in K:
        sample = flights[:10]
        FA_k[k] = sample
        for s in sample:
            AK_f[s] = k
            flights.remove(s)

# Set of flights f which depart from airport k
FD_k = {}
for k in K:
    FD_k[k] = set()

for f in F:
    found = False
    if not found:
        for p in P:
            if f not in p:
                continue
            if f == p[0]:
                FD_k[random.randint(0, num_airports-1)].add(f)
            else:
                FD_k[AK_f[p[p.index(f) - 1]]].add(f)
            found = True

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
oc = [[random.randrange(0, 50, 5) for _ in F] for _ in T]

# Delay cost per minute of arrival delay of flight f
dc = [100 for _ in F]

# Number of passengers in fare class v that are originally scheduled to
# take itinerary p
n = [[random.randint(50, 300) for _ in P] for _ in Y]

# Seating capacity of tail t in T
q = [100 for _ in T]

# Reaccommodation Cost for a passenger reassigned from p to pd.
rc = [[(lambda p, pd: 0 if p == pd else 0.5)(p, pd) for p in P] for pd in P]

# Phantom rate for passenger in fare class v reassigned from p to pd with delay level
# zeta
theta = [[[[0 for _ in Y] for _ in P] for _ in P] for _ in Z]

# Starting location of planes (binary) ((for t) for k)
tb = [random.sample([1] + [0 for _ in range(len(K) - 1)], len(K)) for _ in T]
print(len(tb))

# Capacity of arrival and departure slots
scA = [1 for _ in range(300)]
scD = [1 for _ in range(300)]
