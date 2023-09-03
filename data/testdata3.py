# Initial Data Setup
#     - Both planes have 100 passengers (and a capacity of 100 passengers)
#     - There is only 1 fare class
#     - only one delay level (shouldn't be any delays)
#     - Operating cost of flights = 1
#     - Phantom rate currently set to 0
#     - Each capacity and arrival slot has capacity of 2

#     - Currently no rescheduling has to occur

# ITINERARIES
# T0: F0 (depart A0, arrive A1) -> F1 T1: (depart A1, arrive A2)
# T2: F2 (depart A0, arrive A1) -> F3 T3: (depart A1, arrive A2)

# The scheduled arrival/departure of these planes can be seen below

num_flights = 4
num_tails = 4
num_airports = 3
num_fare_classes = 1
num_delay_levels = 1

# Sets
T = range(num_tails)
F = range(num_flights)
P = [[0, 1], [2, 3]]  # Set of itineraries
K = range(num_airports)
Y = range(num_fare_classes)
Z = range(num_delay_levels)

# Scheduled arrival (departure) time for flight f in F
std = [1, 2, 9.5, 10.5]
sta = [6, 7, 13, 13.5]

# Arrival and Depature slots
AA = [[5, 7], [6, 8], [12, 14], [13, 14]]
DA = [[0, 2], [1, 3], [9, 10], [10, 11]]

# Set of arrival and departure slots compatible with flight f
AAF = [[0], [1], [2], [3]]
DAF = [[0], [1], [2], [3]]

# Set of flights compatible with arrive/departure slot asl/dsl
FAA = [[0], [1], [2], [3]]
FDA = [[0], [1], [2], [3]]

# set of flights compatible with tail T
F_t = [[0, 2], [0, 2], [1, 3], [1, 3]]

# set of tails compatible with flight F
T_f = [[0, 1], [2, 3], [0, 1], [2, 3]]

# Set of flights compatible with a connection from flight f
CF_f = [[1, 3], [], [1, 3], []]

# Set of flights f which arrive to airport k
FA_k = [[], [0, 2], [1, 3]]

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
q = [100, 100, 100, 100]

# Reaccommodation Cost for a passenger reassigned from p to pd.
rc = [[(lambda p, pd: 0 if p == pd else 0.5)(p, pd) for p in P] for pd in P]

# Phantom rate for passenger in fare class v reassigned from p to pd with delay level
# zeta
theta = [[[[0 for _ in Y] for _ in range(len(P))] for _ in range(len(P))] for _ in Z]

# Starting location of planes (binary)
tb = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]

# Capacity of arrival and departure slots
scA = [2, 2, 2, 2]
scD = [2, 2, 2, 2]

# Scheduled buffer time for each flight (set to 0 for now)
sb = [0 for _ in num_flights]
