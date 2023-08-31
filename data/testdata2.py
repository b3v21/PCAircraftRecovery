import random

random.seed(5)
num_flights = 100
num_tails = 20
num_airports = 8
num_fare_classes = 2
num_delay_levels = 2

# Sets
T = range(num_tails)
F = range(num_flights)
P = [[0, 1], [2, 3], [5, 4, 6], [7,8],[9,10],[15,11],[12,13],[14,17,16],[18,19],[20],[]]  # Set of itineraries
K = range(num_airports)
Y = range(num_fare_classes)
Z = range(num_delay_levels)

# Scheduled arrival (departure) time for flight f in F
std = [random.randint(0,20) for _ in F]
sta = [std[i]+round(random.uniform(0,4),2) for i in F]

# Arrival and Depature slots
AA = [[random.randint(0,21), _] for _ in range(90)]
for aa in AA:
    aa[1] = aa[0]+round(random.uniform(0,3),2)
    
DA = [[random.randint(0,21), _] for _ in range(90)]
for da in DA:
    da[1] = da[0]+round(random.uniform(0,3),2)

# Set of arrival and departure slots compatible with flight f
AAF = [[i for i,slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]] for f in F]
DAF = [[i for i,slot in enumerate(AA) if std[f] <= slot[1] and std[f] >= slot[0]] for f in F]

# Set of flights compatible with arrive/departure slot asl/dsl
FAA = [[f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA]
FDA = [[f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA]

# set of flights compatible with tail T
F_t = [random.sample(F, random.randint(0,5)) for _ in T]

# set of tails compatible with flight F
T_f = [[t for t in T if f in F_t[t]] for f in F]

# Set of flights compatible with a connection from flight f
CF_f = [[1], [], [3], []]

# Set of flights f which arrive to airport k
flights = [f for f in F]
FA_k = []
while flights:
    for k in K:
        sample = random.sample(flights,random.randint(0,min(8,len(flights))))
        FA_k += [sample]
        for s in sample:
            flights.remove(s)

# Set of flights f which depart from airport k
FD_k = [[0, 2], [1, 3], []]

# Subset of itineraries compatible with a reasignment from an original itinerary p.
CO_p = [[0], [1]]

# Data

# Cost of operating flight f with tail t
oc = [[1 for _ in range(num_flights)] for _ in range(num_tails)]

# Delay cost per minute of arrival delay of flight f
dc = [100 for _ in range(num_flights)]

# Number of passengers in fare class v that are originally scheduled to
# take itinerary p
n = [[100, 100]]

# Seating capacity of tail t in T
q = [100, 100]

# Reaccommodation Cost for a passenger reassigned from p to pd.
rc = [[(lambda p, pd: 0 if p == pd else 0.5)(p, pd) for p in P] for pd in P]

# Phantom rate for passenger in fare class v reassigned from p to pd with delay level
# zeta
theta = [
    [[[0 for _ in Y] for _ in range(len(P))] for _ in range(len(P))] for _ in Z
]

# Starting location of planes (binary)
tb = [[1, 0, 0], [1, 0, 0]]

# Capacity of arrival and departure slots
scA = [1, 1, 1, 1]
scD = [1, 1, 1, 1]