from gurobipy import *
#from data.test_phantom_passengers_delay_levels import *

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
                        
                        (T0) F0 (depart A0, arrive A1)   ITINERARY: 0
Goal Reassignment:      (T1) F1 (depart A0, arrive A1)   ITINERARY: 1

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
T = range(num_tails) # Set of tails
F = range(num_flights) # Set of flights
P = [[0, 1, 3],[0, 2, 4]] # Set of itineraries
K = range(num_airports) # Set of airports
Y = range(num_fare_classes) # Set of fare classes
Z = range(num_delay_levels) # Set of delay levels

# Scheduled arrival (departure) time for flight f in F
std = [1, 86, 106, 136, 146]
sta = [61, 116, 126, 196, 206]

# Arrival and Depature slots
DA = [(0,15),(81,96),(101,116),(131,146),(141,156)]
AA = [(56,71),(111,126),(121,136),(191,206),(201,216)] 

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
FA_k = {0:[], 1:[0], 2:[2], 3:[1], 4:[4]}

# Airport that flight f arrives at (this isnt actually data in the paper)
AK_f = {0:1, 1:3, 2:2, 3:4, 4:4} 

# Set of flights f which depart from airport k
FD_k = {0:[0], 1:[1,2], 2:[4], 3:[3], 4:[]}

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


########
# Data #
########

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

alpha_regr = [0, 0, 0, 0.290, 0.473]

beta_regr = [0, 0, 0, -0.016, -0.028]

# Phantom rate for passenger in fare class v reassigned from p to pd with delay level
# zeta
theta = [[[[alpha_regr[z] + beta_regr[z]*v for z in Z] for p_from in P] for p_to in P] for v in Y]
#print(theta)

# Starting location of planes (binary) ((for t) for k)
t0 = [[1, 0, 0, 0, 0]]
t1 = [[0, 1, 0, 0, 0]]
t2 = [[0, 1, 0, 0, 0]]
t3 = [[0, 0, 0, 1, 0]]
t4 = [[0, 0, 1, 0, 0]]

# num of tails = 4
# num of airports = 5
tb = []
tb += t0
tb += t1
tb += t2
tb += t3
tb += t4
#print(tb)

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
CF_p = [[(0,1),(1,3)],[(0,2),(2,4)]]

# One if flight f is the last flight of itinerary p, and zero otherwise.
lf = [[0,0],[0,0],[1,0],[0,1],[0,0]]

# Upper bound on the delay, expressed in minutes, corresponding to delay level ζ.
small_theta = [59, 119, 179, 239, 1000]

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

# Kappa = 2000 as in paper
kappa = 2000

# One if flight f was originally scheduled to be operated by tail t, and zero otherwise.
x_hat = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]




BIG_M = 999999999

# QUESTIONS: is rho redundant?


def run_aircraft_recovery() -> None:
    """
    basic initial implementation of Passenger-Centric Integrated Airline Schedule and
    Aircraft Recovery.

    Uses data imported from the file imported above.
    """

    m = Model("airline recovery basic")
    variables = generate_variables(m)
    x, _, _, _, _, _, h, _, _, deltaA, _, _, _, gamma, _, beta = variables

    m.setObjective(
        (
            quicksum(oc[t][f] * x[t, f] for t in T for f in F_t[t])
            + quicksum(fc[f] * gamma[f] for f in F)
            + quicksum(dc[f] * deltaA[f] for f in F)
            + quicksum(
                rc[p][pd]
                * (
                    h[p, pd, v]
                    - quicksum(theta[v][p][pd][g] * beta[v, p, pd, g] for g in Z)
                )
                for v in Y
                for p in range(len(P))
                for pd in CO_p[p]
            )
            + quicksum(
                pc[p][pd][g] * beta[v, p, pd, g]
                for v in Y
                for p in range(len(P))
                for pd in CO_p[p]
                for g in Z
            )
            + kappa
            * quicksum(
                x[t, f] - 2 * x_hat[t][f] * x[t, f] + x_hat[t][f]
                for t in T
                for f in F_t[t]
            )
        ),
        GRB.MINIMIZE,
    )

    flight_scheduling_constraints(m, variables)
    sequencing_and_fleet_size_constraints(m, variables)
    passenger_flow_constraints(m, variables)
    airport_slot_constraints(m, variables)
    flight_delay_constraints(m, variables)
    itinerary_feasibility_constraints(m, variables)
    itinerary_delay_constraints(m, variables)
    beta_linearizing_constraints(m, variables)

    m.optimize()

    generate_output(m, variables)


def generate_variables(m: Model) -> list[dict[list[int], Var]]:
    """
    Generate variables for the model
    """
    # Variables
    # 1 if tail t is assigned to flight f
    x = {(t, f): m.addVar(vtype=GRB.BINARY) for t in T for f in F}

    # 1 if flight f is cancelled
    z = {f: m.addVar(vtype=GRB.BINARY) for f in F}

    #  if flight f is flown and then flight fd is flown
    y = {(f, fd): m.addVar(vtype=GRB.BINARY) for f in F for fd in F if fd != f}

    # 1 if flight f is the last flight in the recovery period operated by its tail
    sigma = {f: m.addVar(vtype=GRB.BINARY) for f in F}

    #  if flight f is the first flight in the recovery period operated by its tail
    rho = {f: m.addVar(vtype=GRB.BINARY) for f in F}

    # 1 if flight f is the first flight for tail t in the recovery period
    phi = {(t, f): m.addVar(vtype=GRB.BINARY) for f in F for t in T}

    # number of passengers in fare class v, reassigned from itinerary p to itinerary p
    h = {
        (p, pd, v): m.addVar() for v in Y for p in range(len(P)) for pd in range(len(P))
    }

    # lambd[p] = 1 if itinerary p is disrupted
    lambd = {p: m.addVar(vtype=GRB.BINARY) for p in range(len(P))}

    # 1 if itinerary p is reassigned to itinerary pd with delay level g
    alpha = {
        (p, pd, g): m.addVar(vtype=GRB.BINARY)
        for p in range(len(P))
        for pd in range(len(P))
        for g in Z
    }

    # arrival delay of flight f
    deltaA = {f: m.addVar() for f in F}

    # departure delay of flight f
    deltaD = {f: m.addVar() for f in F}

    # 1 if arrival slot asl is assigned to flight f
    vA = {(asl, f): m.addVar(vtype=GRB.BINARY) for f in F for asl in range(len(AA))}

    # 1 if departure slot dsl is assigned to flight f
    vD = {(dsl, f): m.addVar(vtype=GRB.BINARY) for f in F for dsl in range(len(DA))}

    # delay absorbed by flight f
    gamma = {f: m.addVar() for f in F}

    # Arrival delay of itinerary pd with respect to planned arrival time of itinerary p
    tao = {(p, pd): m.addVar(lb=-GRB.INFINITY) for p in range(len(P)) for pd in CO_p[p]}

    # number of passengers in fare class v with originally scheduled itinerary p,
    # reassigned to itinerary pd, corresponding to an arrival delay level ζ
    beta = {
        (v, p, pd, g): m.addVar(vtype=GRB.INTEGER)
        for v in Y
        for p in range(len(P))
        for pd in CO_p[p]
        for g in Z
    }

    return [
        x,
        z,
        y,
        sigma,
        rho,
        phi,
        h,
        lambd,
        alpha,
        deltaA,
        deltaD,
        vA,
        vD,
        gamma,
        tao,
        beta,
    ]


def flight_scheduling_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Flight Scheduling Constraints
    """

    x, z, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = variables

    # Every flight must be either flown using exactly one aircraft or must be cancelled
    fsc_1 = {f: m.addConstr(quicksum(x[t, f] for t in T_f[f]) + z[f] == 1) for f in F}


def sequencing_and_fleet_size_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Sequencing & Fleet Size Constraints

    NOTE: Sequencing & Fleet Size Constraints still allow for a tail to not be used
    at all during the recovery period
    """

    x, z, y, sigma, rho, phi, _, _, _, _, _, _, _, _, _, _ = variables

    # Each non-cancelled flight has another flight after it operated by the same tail,
    # unless it is the last flight in the recovery period operated by that tail
    sfsc_1 = {
        f: m.addConstr(quicksum(y[f, fd] for fd in CF_f[f]) + sigma[f] == 1 - z[f])
        for f in F
    }

    # Each non-cancelled flight has another flight before it operated by the same tail,
    # unless it is the first flight in the recovery period operated by that tail
    sfsc_2 = {
        fd: m.addConstr(
            quicksum(y[f, fd] for f in F if fd in CF_f[f]) + rho[fd] == 1 - z[fd]
        )
        for fd in F
    }

    # Consecutive flights operated by the same tail are assigned the same tail

    # Can't assign tail t to flight f and fly f & fd consecutively by tail t without
    # assigning tail t to flight fd
    sfsc_3 = {
        (f, fd, t): m.addConstr(1 + x[t, fd] >= x[t, f] + y[f, fd])
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(T_f[fd]))
    }

    sfsc_4 = {
        (f, fd, t): m.addConstr(1 + x[t, f] >= x[t, fd] + y[f, fd])
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(T_f[fd]))
    }

    # If a flight is assigned a tail and it is the first flight in a sequence of flights,
    # it must be the first flight of that tail
    sfsc_5 = {
        (t, f): m.addConstr(rho[f] + x[t, f] <= 1 + phi[t, f])
        for f in F
        for t in T_f[f]
    }

    # A flight chosen to be the first flight in a sequence of flights can only be assigned
    # to tails whose initial location matches the flight's departure airport
    sfsc_6 = {
        (t, k): m.addConstr(
            quicksum(phi[t, f] for f in list(set(F_t[t]).intersection(FD_k[k])))
            <= tb[t][k]
        )
        for t in T
        for k in K
    }


def passenger_flow_constraints(m: Model, variables: list[dict[list[int], Var]]) -> None:
    """
    Passenger Flow Constraints
    """

    x, _, _, _, _, _, h, lambd, _, _, _, _, _, _, _, beta = variables

    # All passengers are reassigned to some itinerary, which might be the same or
    # different from their originally scheduled itinerary (this include a null itinerary
    # if reassignment is not possible during the recovery period)
    pfc_1 = {
        (p, v): m.addConstr(quicksum(h[p, pd, v] for pd in CO_p[p]) == n[v][p])
        for p in range(len(P))
        for v in Y
    }

    # Passengers can be reassigned only to non-disrupted itineraries
    pfc_2 = {
        (p, v, pd): m.addConstr(h[p, pd, v] <= (1 - lambd[pd]) * n[v][p])
        for p in range(len(P))
        for v in Y
        for pd in CO_p[p]
    }

    # The number of passengers that do show up for their reassigned itineraries does not
    # exceed the total passenger capacity for their reassigned flight
    pfc_3 = {
        f: m.addConstr(
            quicksum(q[t] * x[t, f] for t in T_f[f])
            >= quicksum(
                (
                    h[p, pd, v]
                    - quicksum(theta[v][p][pd][g] * beta[v, p, pd, g] for g in Z)
                )
                for v in Y
                for p in range(len(P))
                for pd in CO_p[p]
                if f in P[pd]
            )
        )
        for f in F
    }


def airport_slot_constraints(m: Model, variables: list[dict[list[int], Var]]) -> None:
    """
    Airport slot constraints
    """

    _, z, _, _, _, _, _, _, _, deltaA, deltaD, vA, vD, _, _, _ = variables

    # Start time of arrival slot asl is no later than the combined scheduled arrival time
    # and arrival delay of flight f, only if the arrival slot is assigned to flight f.
    asc_1 = {
        (f, asl): m.addConstr(
            AA[asl][0] <= sta[f] + deltaA[f] + BIG_M * (1 - vA[asl, f])
        )
        for f in F
        for asl in AAF[f]
    }

    # End time of arrival slot asl is no earlier than the combined scheduled arrival time
    # and arrival delay of flight f, only if the arrival slot is assigned to flight f.
    asc_2 = {
        (f, asl): m.addConstr(
            AA[asl][1] >= sta[f] + deltaA[f] - BIG_M * (1 - vA[asl, f])
        )
        for f in F
        for asl in AAF[f]
    }

    # Each non-cancelled flight is assigned exactly one arrival slot
    asc_3 = {
        f: m.addConstr(quicksum(vA[asl, f] for asl in AAF[f]) == 1 - z[f]) for f in F
    }

    # Arrival slot capacity limit
    asc_4 = {
        asl: m.addConstr(quicksum(vA[asl, f] for f in FAA[asl]) <= scA[asl])
        for asl in range(len(AA))
    }

    # Start time of departure slot asl is no later than the combined scheduled departure
    # time and departure delay of flight f, only if the departure slot is assigned to
    # flight f.
    asc_5 = {
        (f, dsl): m.addConstr(
            DA[dsl][0] <= std[f] + deltaD[f] + BIG_M * (1 - vD[dsl, f])
        )
        for f in F
        for dsl in DAF[f]
    }

    # End time of departure slot asl is no earlier than the combined scheduled departure
    # time and departure delay of flight f, only if the departure slot is assigned to
    # flight f.
    asc_6 = {
        (f, dsl): m.addConstr(
            DA[dsl][1] >= std[f] + deltaD[f] - BIG_M * (1 - vD[dsl, f])
        )
        for f in F
        for dsl in DAF[f]
    }

    # Each non-cancelled flight is assigned exactly one departure slot
    asc_7 = {
        f: m.addConstr(quicksum(vD[dsl, f] for dsl in DAF[f]) == 1 - z[f]) for f in F
    }

    # Departure slot capacity limit
    asc_8 = {
        dsl: m.addConstr(quicksum(vD[dsl, f] for f in FDA[dsl]) <= scD[dsl])
        for dsl in range(len(DA))
    }


def flight_delay_constraints(m: Model, variables: list[dict[list[int], Var]]) -> None:
    """
    Flight Delay Constraints
    """

    x, _, y, _, _, _, _, _, _, deltaA, deltaD, _, _, gamma, _, _ = variables

    # relate the departure and arrival delays of each flight via delay absorption through
    # increased cruise speed.
    fdc_1 = {f: m.addConstr(deltaA[f] >= deltaD[f] - gamma[f] - sb[f]) for f in F}

    # relate the arrival delay of one flight to the departure delay of the next flight
    # operated by the same tail, by accounting for delay propagation.

    fdc_2 = {
        (f, fd, t): m.addConstr(
            deltaD[fd]
            >= deltaA[f]
            + mtt[f][fd][t]
            - ct[f][fd]
            - BIG_M * (3 - x[t, f] - x[t, fd] - y[f, fd])
        )
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(T_f[fd]))
    }


def itinerary_feasibility_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Itinerary Feasibility Constraints: Determine when an itinerary gets disrupted due
    to flight cancelations and due to flight retiming decisions, respectively.
    """

    _, z, _, _, _, _, _, lambd, _, deltaA, deltaD, _, _, _, _, _ = variables

    ifc_1 = {
        (f, p): m.addConstr(lambd[p] >= z[f])
        for f in F
        for p in range(len(P))
        if f in P[p]
    }

    ifc_2 = {
        p: m.addConstr(
            std[pair[1]] + deltaD[pair[1]] - sta[pair[0]] - deltaA[pair[0]]
            >= mct[pair[0]][pair[1]][p] - BIG_M * lambd[p]
        )
        for p in range(len(P)) for pair in CF_p[p]
    }


def itinerary_delay_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Itinerary Delay Constraints
    """

    _, _, _, _, _, _, _, _, alpha, deltaA, _, _, _, _, tao, _ = variables

    # Calculate the passenger arrival delay after the itinerary reassignment.
    idc_1 = {
        (p, pd): m.addConstr(
            tao[p, pd]
            == quicksum(lf[fd][pd] * (deltaA[fd] + sta[fd]) for fd in F)
            - quicksum(lf[f][p] * sta[f] for f in F)
        )
        for p in range(len(P))
        for pd in CO_p[p]
    }

    # Determine the passenger delay level for each pair of compatible itineraries,
    # depending on the actual delay value in minutes.
    idc_2 = {
        (p, pd): m.addConstr(
            tao[p, pd] <= quicksum(small_theta[g] * alpha[p, pd, g] for g in Z)
        )
        for p in range(len(P))
        for pd in CO_p[p]
    }
    idc_3 = {
        (p, pd): m.addConstr(quicksum(alpha[p, pd, g] for g in Z) == 1)
        for p in range(len(P))
        for pd in CO_p[p]
    }


def beta_linearizing_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Constraints which allow beta to behave like alpha * h, while being linear
    """

    _, _, _, _, _, _, h, _, alpha, _, _, _, _, _, _, beta = variables

    blc_1 = {
        (v, p, pd, g): m.addConstr(
            beta[v, p, pd, g] <= h[p, pd, v] + BIG_M * (1 - alpha[p, pd, g])
        )
        for v in Y
        for p in range(len(P))
        for pd in CO_p[p]
        for g in Z
    }
    blc_2 = {
        (v, p, pd, g): m.addConstr(
            beta[v, p, pd, g] >= h[p, pd, v] - BIG_M * (1 - alpha[p, pd, g])
        )
        for v in Y
        for p in range(len(P))
        for pd in CO_p[p]
        for g in Z
    }
    blc_3 = {
        (v, p, pd, g): m.addConstr(beta[v, p, pd, g] <= BIG_M * alpha[p, pd, g])
        for v in Y
        for p in range(len(P))
        for pd in CO_p[p]
        for g in Z
    }


def generate_output(m: Model, variables: list[dict[list[int], Var]]) -> None:
    x, z, y, sigma, rho, phi, h, lambd, _, _, _, vA, vD, _, _, _ = variables
    print(72 * "-")
    print("\nTail to Flight Assignments:")
    for f in F:
        found_chained_flight = False
        for fd in CF_f[f]:
            if fd != f:
                if y[f, fd].x > 0.9:
                    for t in T:
                        if x[t, f].x > 0.9 and x[t, fd].x > 0.9:
                            found_chained_flight = True
                            print(f"T{t}: F{f} -> F{fd}")
                            if found_chained_flight:
                                break
                    if found_chained_flight:
                        break
        if not found_chained_flight:
            for t in T:
                if x[t, f].x > 0.9 and (sigma[f].x < 0.9 or rho[f].x > 0.9):
                    print(f"T{t}: F{f}")

    print("\nCancelled Flights:")
    for f in F:
        if z[f].x > 0.9:
            print(f"F{f} cancelled")

    print("\nDeparture / arrival slots:")
    for f in F:
        for dsl in range(len(DA)):
            for asl in range(len(AA)):
                if vD[dsl, f].x > 0.9 and vA[asl, f].x > 0.9:
                    print(
                        f"F{f}: \t Departure Slot: {DA[dsl]} \t Arrival Slot: {AA[asl]}"
                    )

    print("\nDisrupted Itineraries:")
    for p in range(len(P)):
        if lambd[p].x > 0.9:
            print(f"I{p} disrupted:")

            for pd in range(len(P)):
                for v in Y:
                    if p != pd:
                        print(
                            f"    I{p} -> I{pd} (fare class: {v}) people: {int(h[p, pd, v].x)}"
                        )

    print("\n" + 72 * "-")


if __name__ == "__main__":
    run_aircraft_recovery()
