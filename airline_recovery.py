from gurobipy import *

BIG_M = 999999999


def generate_variables(
    m: Model,
    T,
    F,
    Y,
    Z,
    P,
    AA,
    DA,
    CO_p,
) -> list[dict[list[int], Var]]:
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
    h = {(P.index(p), P.index(pd), v): m.addVar() for v in Y for p in P for pd in P}

    # lambd[p] = 1 if itinerary p is disrupted
    lambd = {P.index(p): m.addVar(vtype=GRB.BINARY) for p in P}

    # 1 if itinerary p is reassigned to itinerary pd with delay level g
    alpha = {
        (P.index(p), P.index(pd), g): m.addVar(vtype=GRB.BINARY)
        for p in P
        for pd in P
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
    tao = {
        (P.index(p), pd): m.addVar(lb=-GRB.INFINITY)
        for p in P
        for pd in CO_p[P.index(p)]
    }

    # number of passengers in fare class v with originally scheduled itinerary p,
    # reassigned to itinerary pd, corresponding to an arrival delay level ζ
    beta = {
        (v, P.index(p), pd, g): m.addVar(vtype=GRB.INTEGER)
        for v in Y
        for p in P
        for pd in CO_p[P.index(p)]
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


def set_objective(
    m: Model,
    variables: list[dict[list[int], Var]],
    T,
    F,
    Y,
    Z,
    P,
    F_t,
    CO_p,
    oc,
    dc,
    rc,
    theta,
    fc,
    pc,
    kappa,
    x_hat,
) -> None:
    """
    Set the objective function for the model
    """

    x, _, _, _, _, _, h, _, _, deltaA, _, _, _, gamma, _, beta = variables

    print("setting objective...")
    m.setObjective(
        (
            quicksum(oc[t, f] * x[t, f] for t in T for f in F_t[t])
            + quicksum(fc[f] * gamma[f] for f in F)
            + quicksum(dc[f] * deltaA[f] for f in F)
            + quicksum(
                rc[P.index(p), pd]
                * (
                    h[P.index(p), pd, v]
                    - quicksum(
                        theta[v, P.index(p), pd, g] * beta[v, P.index(p), pd, g]
                        for g in Z
                    )
                )
                for v in Y
                for p in P
                for pd in CO_p[P.index(p)]
            )
            + quicksum(
                pc[g, P.index(p), pd] * beta[v, P.index(p), pd, g]
                for v in Y
                for p in P
                for pd in CO_p[P.index(p)]
                for g in Z
            )
            + kappa
            * quicksum(
                x[t, f] - 2 * x_hat[f, t] * x[t, f] + x_hat[f, t]
                for t in T
                for f in F_t[t]
            )
        ),
        GRB.MINIMIZE,
    )


def flight_scheduling_constraints(
    m: Model, variables: list[dict[list[int], Var]], F, T_f
) -> None:
    """
    Flight Scheduling Constraints
    """

    x, z, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = variables

    print("adding flight scheduling constraints...")

    # Every flight must be either flown using exactly one aircraft or must be cancelled
    fsc_1 = {f: m.addConstr(quicksum(x[t, f] for t in T_f[f]) + z[f] == 1) for f in F}


def sequencing_and_fleet_size_constraints(
    m: Model, variables: list[dict[list[int], Var]], T, F, K, F_t, T_f, FD_k, CF_f, tb
) -> None:
    """
    Sequencing & Fleet Size Constraints

    NOTE: Sequencing & Fleet Size Constraints still allow for a tail to not be used
    at all during the recovery period
    """

    x, z, y, sigma, rho, phi, _, _, _, _, _, _, _, _, _, _ = variables

    print("adding sequencing and fleet size constraints... ")

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
            <= tb[t, k]
        )
        for t in T
        for k in K
    }


def passenger_flow_constraints(
    m: Model, variables: list[dict[list[int], Var]], F, Y, Z, P, T_f, CO_p, theta, n, q
) -> None:
    """
    Passenger Flow Constraints
    """

    x, _, _, _, _, _, h, lambd, _, _, _, _, _, _, _, beta = variables

    print("adding passenger flow constraints...")

    # All passengers are reassigned to some itinerary, which might be the same or
    # different from their originally scheduled itinerary (this include a null itinerary
    # if reassignment is not possible during the recovery period)
    pfc_1 = {
        (P.index(p), v): m.addConstr(
            quicksum(h[P.index(p), pd, v] for pd in CO_p[P.index(p)])
            == n[v, P.index(p)]
        )
        for p in P
        for v in Y
    }

    # Passengers can be reassigned only to non-disrupted itineraries
    pfc_2 = {
        (P.index(p), v, pd): m.addConstr(
            h[P.index(p), pd, v] <= (1 - lambd[pd]) * n[v, P.index(p)]
        )
        for p in P
        for v in Y
        for pd in CO_p[P.index(p)]
    }

    # The number of passengers that do show up for their reassigned itineraries does not
    # exceed the total passenger capacity for their reassigned flight
    pfc_3 = {
        f: m.addConstr(
            quicksum(q[t] * x[t, f] for t in T_f[f])
            >= quicksum(
                (
                    h[P.index(p), pd, v]
                    - quicksum(
                        theta[v, P.index(p), pd, g] * beta[v, P.index(p), pd, g]
                        for g in Z
                    )
                )
                for v in Y
                for p in P
                for pd in CO_p[P.index(p)]
                if f in P[pd]
            )
        )
        for f in F
    }


def airport_slot_constraints(
    m: Model,
    variables: list[dict[list[int], Var]],
    F,
    Z,
    sta,
    std,
    AA,
    DA,
    AAF,
    DAF,
    FAA,
    FDA,
    scA,
    scD,
) -> None:
    """
    Airport slot constraints
    """

    _, z, _, _, _, _, _, _, _, deltaA, deltaD, vA, vD, _, _, _ = variables

    print("adding airport slot constraints...")

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
        asl: m.addConstr(quicksum(vA[AA.index(asl), f] for f in FAA[asl]) <= scA[asl])
        for asl in AA
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

    # # Departure slot capacity limit
    asc_8 = {
        dsl: m.addConstr(quicksum(vD[DA.index(dsl), f] for f in FDA[dsl]) <= scD[dsl])
        for dsl in DA
    }


def flight_delay_constraints(
    m: Model,
    variables: list[dict[list[int], Var]],
    T,
    F,
    T_f,
    CF_f,
    sb,
    mtt,
    ct,
) -> None:
    """
    Flight Delay Constraints
    """

    x, _, y, _, _, _, _, _, _, deltaA, deltaD, _, _, gamma, _, _ = variables

    print("adding flight delay constraints...")

    # relate the departure and arrival delays of each flight via delay absorption through
    # increased cruise speed.
    fdc_1 = {f: m.addConstr(deltaA[f] >= deltaD[f] - gamma[f] - sb[f]) for f in F}

    # relate the arrival delay of one flight to the departure delay of the next flight
    # operated by the same tail, by accounting for delay propagation.

    fdc_2 = {
        (f, fd, t): m.addConstr(
            deltaD[fd]
            >= deltaA[f]
            + mtt[t, f, fd]
            - ct[f, fd]
            - BIG_M * (3 - x[t, f] - x[t, fd] - y[f, fd])
        )
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(set(T_f[fd])))
    }


def itinerary_feasibility_constraints(
    m: Model,
    variables: list[dict[list[int], Var]],
    F,
    P,
    sta,
    std,
    CF_p,
    mct,
) -> None:
    """
    Itinerary Feasibility Constraints: Determine when an itinerary gets disrupted due
    to flight cancelations and due to flight retiming decisions, respectively.
    """

    _, z, _, _, _, _, _, lambd, _, deltaA, deltaD, _, _, _, _, _ = variables

    print("adding itinerary feasibility constraints...")

    ifc_1 = {
        (f, p): m.addConstr(lambd[p] >= z[f])
        for f in F
        for p in range(len(P))
        if f in P[p]
    }

    ifc_2 = {
        (P.index(p), (f, fd)): m.addConstr(
            std[fd] + deltaD[fd] - sta[f] - deltaA[f]
            >= mct[P.index(p), f, fd] - BIG_M * lambd[P.index(p)]
        )
        for p in P
        for (f, fd) in CF_p[P.index(p)]
        if CF_p[P.index(p)] != []
    }


def itinerary_delay_constraints(
    m: Model,
    variables: list[dict[list[int], Var]],
    F,
    Z,
    P,
    sta,
    CO_p,
    lf,
    small_theta,
) -> None:
    """
    Itinerary Delay Constraints
    """

    _, _, _, _, _, _, _, _, alpha, deltaA, _, _, _, _, tao, _ = variables

    print("adding itinerary delay constraints...")

    # Calculate the passenger arrival delay after the itinerary reassignment.
    idc_1 = {
        (P.index(p), pd): m.addConstr(
            tao[P.index(p), pd]
            == quicksum(lf[pd, fd] * (deltaA[fd] + sta[fd]) for fd in F)
            - quicksum(lf[P.index(p), f] * sta[f] for f in F)
        )
        for p in P
        for pd in CO_p[P.index(p)]
    }

    # Determine the passenger delay level for each pair of compatible itineraries,
    # depending on the actual delay value in minutes.
    idc_2 = {
        (P.index(p), pd): m.addConstr(
            tao[P.index(p), pd]
            <= quicksum(small_theta[g] * alpha[P.index(p), pd, g] for g in Z)
        )
        for p in P
        for pd in CO_p[P.index(p)]
    }

    idc_3 = {
        (P.index(p), pd): m.addConstr(
            quicksum(alpha[P.index(p), pd, g] for g in Z) == 1
        )
        for p in P
        for pd in CO_p[P.index(p)]
    }


def beta_linearizing_constraints(
    m: Model,
    variables: list[dict[list[int], Var]],
    Y,
    Z,
    P,
    CO_p,
) -> None:
    """
    Constraints which allow beta to behave like alpha * h, while being linear
    """

    _, _, _, _, _, _, h, _, alpha, _, _, _, _, _, _, beta = variables

    print("adding beta linearizing constraints...")

    blc_1 = {
        (v, P.index(p), pd, g): m.addConstr(
            beta[v, P.index(p), pd, g]
            <= h[P.index(p), pd, v] + BIG_M * (1 - alpha[P.index(p), pd, g])
        )
        for v in Y
        for p in P
        for pd in CO_p[P.index(p)]
        for g in Z
    }

    blc_2 = {
        (v, P.index(p), pd, g): m.addConstr(
            beta[v, P.index(p), pd, g]
            >= h[P.index(p), pd, v] - BIG_M * (1 - alpha[P.index(p), pd, g])
        )
        for v in Y
        for p in P
        for pd in CO_p[P.index(p)]
        for g in Z
    }

    blc_3 = {
        (v, P.index(p), pd, g): m.addConstr(
            beta[v, P.index(p), pd, g] <= BIG_M * alpha[P.index(p), pd, g]
        )
        for v in Y
        for p in P
        for pd in CO_p[P.index(p)]
        for g in Z
    }


def generate_x_hat(m: Model, variables: list[dict[list[int], Var]], F, T):
    """
    Using the x values from the first optimization, generate x_hat values for the
    second optimization
    """

    x, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = variables

    x_hat = {}

    for f in F:
        for t in T:
            x_hat[(f, t)] = int(x[t, f].x)

    return x_hat


def generate_output(
    m: Model,
    variables: list[dict[list[int], Var]],
    T,
    F,
    Y,
    Z,
    P,
    sta,
    std,
    AA,
    DA,
    DK_f,
    AK_f,
    CF_f,
    n,
    fc,
) -> None:
    x, z, y, sigma, rho, phi, h, lambd, _, deltaA, deltaD, vA, vD, _, _, _ = variables

    chained_flights = {}

    print(60 * "-")
    print("\nFlight Information:")
    for f in F:
        found_chained_flight = False
        for fd in CF_f[f]:
            if fd != f:
                if y[f, fd].x > 0.9:
                    for t in T:
                        if x[t, f].x > 0.9 and x[t, fd].x > 0.9:
                            found_chained_flight = True
                            if t in chained_flights:
                                chained_flights[t].append(f)
                            chained_flights[t] = [f, fd]
                            if found_chained_flight:
                                break
                    if found_chained_flight:
                        break
        if not found_chained_flight:
            for t in T:
                if x[t, f].x > 0.9 and (sigma[f].x < 0.9 or rho[f].x > 0.9):
                    for dsl in range(len(DA)):
                        for asl in range(len(AA)):
                            if vD[dsl, f].x > 0.9 and vA[asl, f].x > 0.9:
                                # These flight times include arr/dep delay
                                print(
                                    f"Tail {t}: F{f} \t  \t {DK_f[f]} ({round(std[f] + deltaD[f].x,1)}) -> {AK_f[f]} ({round(sta[f]+deltaA[f].x,1)})\t"
                                )

    for t, cf in chained_flights.items():
        output = ""
        output = f"Tail {t}: "
        for i, f in enumerate(cf):
            output += f"F{f}"
            if i != len(cf) - 1:
                output += " -> "
            else:
                output += "\t "

        for f in cf:
            output += f"{DK_f[f]} ({round(std[f] + deltaD[f].x,1)}) -> {AK_f[f]} ({round(sta[f]+deltaA[f].x,1)}) \t"
        print(output)

    cancelled = False
    cancelled_count = 0
    print("\nCancelled Flights:")
    for f in F:
        if z[f].x > 0.9:
            print(f"F{f} cancelled")
            cancelled = True
            cancelled_count += 1
    if not cancelled:
        print("No Flights Cancelled")

    print("\nTotal Flights Cancelled:", cancelled_count)

    disrupted_itins = False
    disrupted_passengers = 0
    disrupted_count = 0
    print("\nDisrupted Itineraries:")
    for p in P:
        if lambd[P.index(p)].x > 0.9:
            disrupted_itins = True
            disrupted_count += 1
            print(f"I{P.index(p)} disrupted.")

    print("\nTotal Disrupted Itineraries:", disrupted_count)

    if not disrupted_itins:
        print("No Itineraries Disrupted")

    print("\nPassenger Reassignments:")
    for p in P:
        for pd in P:
            for v in Y:
                if p != pd:
                    if int(h[P.index(p), P.index(pd), v].x) > 0:
                        print(
                            f"    I{P.index(p)} {*P[P.index(p)],} -> I{P.index(pd)} {*P[P.index(pd)],} (FC: {v}) people reassigned: {int(h[P.index(p), P.index(pd), v].x)}"
                        )
                        disrupted_passengers += int(h[P.index(p), P.index(pd), v].x)

    total_passengers = sum([n[v, P.index(p)] for v in Y for p in P])
    print(f"\nTotal Reassigned Passengers: {disrupted_passengers}")
    print(
        f"Percentage of Passengers Reassigned: {round(disrupted_passengers/total_passengers*100,2)}%"
    )

    print("\nFlight Delays:")
    for f in F:
        if z[f].x < 0.1:
            out = ""
            if deltaD[f].x > 1e-5:
                out += f"F{f} Departure Delay: {round(deltaD[f].x,1)} "
            if deltaA[f].x > 1e-5:
                out += f"F{f} Arrival Delay: {round(deltaA[f].x,1)} "
            if deltaD[f].x - deltaA[f].x > 1e-3:
                out += f"-> Delay Absorbed: {round(deltaD[f].x - deltaA[f].x,2)}"
            if deltaA[f].x > 1e-5 or deltaD[f].x > 1e-5:
                print(out)

    print("\nTotal Cost: ", round(m.objVal, 2))

    print("\n" + 60 * "-")


if __name__ == "__main__":
    run_aircraft_recovery()
