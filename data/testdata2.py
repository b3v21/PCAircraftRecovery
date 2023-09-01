import random

random.seed(3)
num_flights = 80
num_tails = 20
num_airports = 8
num_fare_classes = 2
num_delay_levels = 2

# Sets
T = range(num_tails)
F = range(num_flights)
P = [
    [0, 1],
    [2, 3],
    [5, 4, 6],
    [7, 8],
    [9, 10],
    [15, 11],
    [12, 13],
    [14, 17, 16],
    [18, 19],
    [20],
    [],
]  # Set of itineraries
K = range(num_airports)
Y = range(num_fare_classes)
Z = range(num_delay_levels)

# Scheduled arrival (departure) time for flight f in F
std = [random.randint(0, 20) for _ in F]
sta = [std[i] + round(random.uniform(0, 4), 2) for i in F]

# Arrival and Depature slots
AA = [[random.randint(0, 21), _] for _ in range(300)]
for aa in AA:
    aa[1] = aa[0] + round(random.uniform(0, 3), 2)

DA = [[random.randint(0, 21), _] for _ in range(300)]
for da in DA:
    da[1] = da[0] + round(random.uniform(0, 3), 2)

# Set of arrival and departure slots compatible with flight f
AAF = [
    [i for i, slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]]
    for f in F
]
DAF = [
    [i for i, slot in enumerate(AA) if std[f] <= slot[1] and std[f] >= slot[0]]
    for f in F
]

# Set of flights compatible with arrive/departure slot asl/dsl
FAA = [[f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA]
FDA = [[f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA]

# set of flights compatible with tail T
F_t = [random.sample(F, random.randint(0, 5)) for _ in T] # Assume this is right for now

# set of tails compatible with flight F
T_f = [[t for t in T if f in F_t[t]] for f in F]

# Set of flights f which arrive to airport k
flights = [f for f in F]
FA_k = {}
AK_f = {}  # Airport that flight f arrives at (this isnt actually data in the paper)

while flights:
    for k in K:
        sample = random.sample(flights, 10) 
        FA_k[k] = sample
        for s in sample:
            AK_f[s] = k
            flights.remove(s)

# Set of flights f which depart from airport k
FD_k = {}
for k in K:
    FD_k[k] = set()
for k in K:
    for f in F:
        for p in P:
            if f not in p:
                continue
            for i, _ in enumerate(p):
                if p[i] == f:
                    FD_k[AK_f[p[i - 1]]].add(f)
                    
import pdb; pdb.set_trace()

# Set of flights compatible with a connection from flight f
CF_f = [[fd for fd in F if FD_k[fd] == FA_k[f] and fd != f] for f in F]

# Subset of itineraries compatible with a reasignment from an original itinerary p.
CO_p = [
    [i for i, p in enumerate(P)] for p in P
]  # This is a bit silly as currently any flight can be rescheduled to any flight

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
