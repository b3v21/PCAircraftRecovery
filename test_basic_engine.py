from gurobipy import *
from airline_recovery import *
from collections import deque

TIME_HORIZON = 72


def test_basic_solve():
    """
    Builds and runs a simulation with the base data and confirms it has the following format.

    T0: F0 (depart A0, arrive A1) -> T0: F1 (depart A1, arrive A2)        P0
    T1: F2 (depart A0, arrive A1) -> T1: F3 (depart A1, arrive A2)        P1
    ...
    TN: F2N (depart A0, arrive A1) -> T4: F(2N+1) (depart A1, arrive A2)  PN

    Departures occuring every 0.5 hrs
    Arrivals occuring every
    """

    basic_solve = Model("test_basic_solve")

    num_flights = 20
    num_tails = 10
    num_airports = 3
    num_fare_classes = 2
    num_delay_levels = 2
    num_time_instances = 8

    T = range(num_tails)
    F = range(num_flights)
    P = [[i, i + 1] for i in range(0, num_flights, 2)]
    K = range(num_airports)
    Y = range(num_fare_classes)
    Z = range(num_delay_levels)

    std = {f: f + 0.5 for f in F}
    sta = {f: f + 1 for f in F}

    # MAINTENANCE DATA / SETS
    PI = range(num_time_instances)
    MO = set([(f, fd) for f in F for fd in F if f != fd])
    F_pi = {
        pi: [f for f in F if sta[f] <= (1 + pi) * (TIME_HORIZON / num_time_instances)]
        for pi in PI
    }
    K_m = set([k for k in K])
    T_m = set([])
    PI_m = set([0, 1])

    abh = {t: 20 for t in T}
    sbh = {f: 20 for f in F}
    mbh = {t: 40 for t in T}
    mt = {t: 0.5 for t in T}
    aw = {k: 1 for k in K}

    DA = [(f, f + 1) for f in F]
    AA = [(f - 0.5, f + 0.5) for f in range(1, num_flights + 1)]

    AAF = {
        f: [i for i, slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]]
        for f in F
    }
    DAF = {
        f: [i for i, slot in enumerate(DA) if std[f] <= slot[1] and std[f] >= slot[0]]
        for f in F
    }

    FAA = {asl: [f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA}
    FDA = {dsl: [f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA}

    F_t = {t: list(F) for t in T}
    T_f = {f: [t for t in T if f in F_t[t]] for f in F}

    flights = [f for f in F]
    FA_k = {}
    for k in K:
        FA_k[k] = []

    AK_f = {}

    for f in F:
        if f % 2 == 0:
            FA_k[1].append(f)
            AK_f[f] = 1
        else:
            FA_k[2].append(f)
            AK_f[f] = 2

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

    CF_f = {
        f: [fd for fd in F if AK_f[f] == DK_f[fd] and sta[f] <= std[fd] and fd != f]
        for f in F
    }

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
    oc = {(t, f): 1500 for f in F for t in T}
    dc = {f: 100 for f in F}
    n = {(v, P.index(p)): 50 for p in P for v in Y}
    q = {t: 100 for t in T}
    rc = {
        (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 0.5)(p, pd)
        for p in P
        for pd in P
    }
    theta = {
        (v, P.index(p), P.index(pd), z): 0 for v in Y for p in P for pd in P for z in Z
    }

    tb = {(t, k): 1 if k == 0 else 0 for t in T for k in K}
    scA = {asl: 1 for asl in AA}
    scD = {dsl: 1 for dsl in DA}
    sb = {f: 0 for f in F}
    mtt = {(t, f, fd): 0 for t in T for f in F for fd in F}
    mct = {(P.index(p), f, fd): 0 for p in P for f in F for fd in F}
    ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}
    CF_p = {P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])] for p in P}
    lf = {(P.index(p), f): 1 if p[-1] == f else 0 for p in P for f in F}
    small_theta = {z: 1000 for z in Z}
    fc = {f: 100 for f in F}
    pc = {(z, P.index(p), P.index(pd)): 0 for z in Z for p in P for pd in P}
    kappa = 100
    x_hat = {(f, t): 1 if t == f or t + 1 == f else 0 for f in F for t in T}

    variables = generate_variables(basic_solve, T, F, Y, Z, P, AA, DA, CO_p, K)
    set_objective(
        basic_solve,
        variables,
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
    )
    flight_scheduling_constraints(basic_solve, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        basic_solve, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        basic_solve, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        basic_solve, variables, F, Z, sta, std, AA, DA, AAF, DAF, FAA, FDA, scA, scD
    )
    flight_delay_constraints(basic_solve, variables, T, F, T_f, CF_f, sb, mtt, ct)
    itinerary_feasibility_constraints(basic_solve, variables, F, P, sta, std, CF_p, mct)
    itinerary_delay_constraints(
        basic_solve, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    beta_linearizing_constraints(basic_solve, variables, Y, Z, P, CO_p)
    maintenance_schedule_constraints(
        basic_solve, variables, T_m, sta, T_f, F, F_t, mt, MO, std
    )
    workshop_schedule_constraints(
        basic_solve, variables, F_t, T_m, K_m, F, T_f, K, aw, FA_k
    )
    maintenance_check_constraints(
        basic_solve, variables, T_m, PI_m, F_pi, sbh, mbh, F_t, MO, abh, F, T_f
    )

    print("optimizing...")
    basic_solve.optimize()

    print("generating output...")
    generate_output(
        basic_solve,
        variables,
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
        T_m,
    )

    (
        _,
        z,
        _,
        _,
        _,
        _,
        _,
        lambd,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = variables

    # No flights cancelled and no itineraries disrupted
    for f in F:
        assert z[f].x < 0.9
    for p in P:
        assert lambd[P.index(p)].x < 0.9


def test_basic_reschedule_if_cheaper_tail():
    """
    Ensures that a flight is rescheduled to a cheaper tail if it is available.
    """

    basic_reschedule_if_cheaper_tail = Model("test_basic_reschedule_if_cheaper_tail")

    num_flights = 1
    num_tails = 2
    num_airports = 2
    num_fare_classes = 2
    num_delay_levels = 2

    T = range(num_tails)
    F = range(num_flights)
    P = [[0]]
    K = range(num_airports)
    Y = range(num_fare_classes)
    Z = range(num_delay_levels)

    std = {f: f + 0.5 for f in F}
    sta = {f: f + 1 for f in F}

    DA = [(f, f + 1) for f in F]
    AA = [(f - 0.5, f + 0.5) for f in range(1, num_flights + 1)]

    AAF = {
        f: [i for i, slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]]
        for f in F
    }
    DAF = {
        f: [i for i, slot in enumerate(DA) if std[f] <= slot[1] and std[f] >= slot[0]]
        for f in F
    }

    FAA = {asl: [f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA}
    FDA = {dsl: [f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA}

    F_t = {0: [0], 1: [0]}
    T_f = {0: [0, 1]}

    flights = [f for f in F]
    FA_k = {}
    for k in K:
        FA_k[k] = []

    AK_f = {}

    for f in F:
        if f % 2 == 0:
            FA_k[1].append(f)
            AK_f[f] = 1
        else:
            FA_k[2].append(f)
            AK_f[f] = 2

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

    CF_f = {0: []}
    CO_p = {0: [0]}

    oc = {(0, 0): 1000000000, (1, 0): 100}
    dc = {f: 100 for f in F}
    n = {(v, P.index(p)): 50 for p in P for v in Y}
    q = {t: 100 for t in T}
    rc = {
        (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 0.5)(p, pd)
        for p in P
        for pd in P
    }
    theta = {
        (v, P.index(p), P.index(pd), z): 0 for v in Y for p in P for pd in P for z in Z
    }
    tb = {(t, k): 1 if k == 0 else 0 for t in T for k in K}
    scA = {asl: 1 for asl in AA}
    scD = {dsl: 1 for dsl in DA}
    sb = {f: 0 for f in F}
    mtt = {(t, f, fd): 0 for t in T for f in F for fd in F}
    mct = {(P.index(p), f, fd): 0 for p in P for f in F for fd in F}
    ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}
    CF_p = {0: []}
    lf = {(0, 0): 1}
    small_theta = {z: 1000 for z in Z}
    fc = {f: 100 for f in F}
    pc = {(z, P.index(p), P.index(pd)): 0 for z in Z for p in P for pd in P}
    kappa = 100
    x_hat = {(f, t): 1 if t == f or t + 1 == f else 0 for f in F for t in T}

    variables = generate_variables(
        basic_reschedule_if_cheaper_tail, T, F, Y, Z, P, AA, DA, CO_p
    )
    set_objective(
        basic_reschedule_if_cheaper_tail,
        variables,
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
    )
    flight_scheduling_constraints(basic_reschedule_if_cheaper_tail, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        basic_reschedule_if_cheaper_tail, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        basic_reschedule_if_cheaper_tail, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        basic_reschedule_if_cheaper_tail,
        variables,
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
    )
    flight_delay_constraints(
        basic_reschedule_if_cheaper_tail, variables, T, F, T_f, CF_f, sb, mtt, ct
    )
    itinerary_feasibility_constraints(
        basic_reschedule_if_cheaper_tail, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        basic_reschedule_if_cheaper_tail, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    beta_linearizing_constraints(
        basic_reschedule_if_cheaper_tail, variables, Y, Z, P, CO_p
    )

    print("optimizing...")
    basic_reschedule_if_cheaper_tail.optimize()

    print("generating output...")
    generate_output(
        basic_reschedule_if_cheaper_tail,
        variables,
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
    )

    x, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = variables

    # Ensure cheaper flight is being used.
    assert x[1, 0].x == 1
    assert x[0, 0].x == 0


def test_basic_reschedule_if_plane_cap_too_small():
    """
    This will test that a flight is flown is reassigned to a cheaper tail

    Reassigns F0: (depart A0, arrive A1) from T0 to T1 due to T0 not having enough seats
    """

    basic_reschedule_if_plane_cap_too_small = Model(
        "test_basic_reschedule_if_plane_cap_too_small"
    )

    num_flights = 2
    num_tails = 2
    num_airports = 3
    num_fare_classes = 2
    num_delay_levels = 2

    T = range(num_tails)
    F = range(num_flights)
    P = [[i, i + 1] for i in range(0, num_flights, 2)]
    K = range(num_airports)
    Y = range(num_fare_classes)
    Z = range(num_delay_levels)

    std = {f: f + 0.5 for f in F}
    sta = {f: f + 1 for f in F}

    DA = [(f, f + 1) for f in F]
    AA = [(f - 0.5, f + 0.5) for f in range(1, num_flights + 1)]

    AAF = {
        f: [i for i, slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]]
        for f in F
    }
    DAF = {
        f: [i for i, slot in enumerate(DA) if std[f] <= slot[1] and std[f] >= slot[0]]
        for f in F
    }

    FAA = {asl: [f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA}
    FDA = {dsl: [f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA}

    F_t = {t: list(F) for t in T}
    T_f = {f: [t for t in T if f in F_t[t]] for f in F}

    flights = [f for f in F]
    FA_k = {}
    for k in K:
        FA_k[k] = []

    AK_f = {}
    for f in F:
        if f % 2 == 0:
            FA_k[1].append(f)
            AK_f[f] = 1
        else:
            FA_k[2].append(f)
            AK_f[f] = 2

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

    CF_f = {
        f: [fd for fd in F if AK_f[f] == DK_f[fd] and sta[f] <= std[fd] and fd != f]
        for f in F
    }
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

    oc = {(t, f): 100 for f in F for t in T}
    dc = {f: 100 for f in F}
    n = {(v, P.index(p)): 50 for p in P for v in Y}
    q = {0: 10, 1: 100}
    rc = {
        (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 0.5)(p, pd)
        for p in P
        for pd in P
    }
    theta = {
        (v, P.index(p), P.index(pd), z): 0 for v in Y for p in P for pd in P for z in Z
    }

    tb = {(t, k): 1 if k == 0 else 0 for t in T for k in K}
    scA = {asl: 1 for asl in AA}
    scD = {dsl: 1 for dsl in DA}
    sb = {f: 0 for f in F}
    mtt = {(t, f, fd): 0 for t in T for f in F for fd in F}
    mct = {(P.index(p), f, fd): 0 for p in P for f in F for fd in F}
    ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}
    CF_p = {P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])] for p in P}
    lf = {(P.index(p), f): 1 if p[-1] == f else 0 for p in P for f in F}
    small_theta = {z: 1000 for z in Z}
    fc = {f: 100 for f in F}
    pc = {(z, P.index(p), P.index(pd)): 0 for z in Z for p in P for pd in P}
    kappa = 10000
    x_hat = {(0, 0): 1, (0, 1): 0, (1, 0): 1, (1, 1): 0}

    variables = generate_variables(
        basic_reschedule_if_plane_cap_too_small, T, F, Y, Z, P, AA, DA, CO_p
    )
    set_objective(
        basic_reschedule_if_plane_cap_too_small,
        variables,
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
    )
    flight_scheduling_constraints(
        basic_reschedule_if_plane_cap_too_small, variables, F, T_f
    )
    sequencing_and_fleet_size_constraints(
        basic_reschedule_if_plane_cap_too_small,
        variables,
        T,
        F,
        K,
        F_t,
        T_f,
        FD_k,
        CF_f,
        tb,
    )
    passenger_flow_constraints(
        basic_reschedule_if_plane_cap_too_small,
        variables,
        F,
        Y,
        Z,
        P,
        T_f,
        CO_p,
        theta,
        n,
        q,
    )
    airport_slot_constraints(
        basic_reschedule_if_plane_cap_too_small,
        variables,
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
    )
    flight_delay_constraints(
        basic_reschedule_if_plane_cap_too_small, variables, T, F, T_f, CF_f, sb, mtt, ct
    )
    itinerary_feasibility_constraints(
        basic_reschedule_if_plane_cap_too_small, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        basic_reschedule_if_plane_cap_too_small,
        variables,
        F,
        Z,
        P,
        sta,
        CO_p,
        lf,
        small_theta,
    )
    beta_linearizing_constraints(
        basic_reschedule_if_plane_cap_too_small, variables, Y, Z, P, CO_p
    )

    print("optimizing...")
    basic_reschedule_if_plane_cap_too_small.optimize()

    print("generating output...")
    generate_output(
        basic_reschedule_if_plane_cap_too_small,
        variables,
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
    )

    x, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = variables

    # Reassigns T0 -> T1 due to lack of capacity on T0.
    assert x[1, 0].x == 1
    assert x[0, 0].x == 0


def test_passenger_itinerary_disruption():
    """
    This is a test for checking passenger reassignment between itineraries

    Original:               (T0) F0 (depart A0, arrive A1)   ITINERARY: 0
    Goal Reassignment:      (T1) F1 (depart A0, arrive A1)   ITINERARY: 1
    """

    passenger_itinerary_disruption = Model("test_passenger_itinerary_disruption")

    num_flights = 2
    num_tails = 2
    num_airports = 2
    num_fare_classes = 2
    num_delay_levels = 2

    T = range(num_tails)
    F = range(num_flights)
    P = [[0], [1]]
    K = range(num_airports)
    Y = range(num_fare_classes)
    Z = range(num_delay_levels)

    std = {f: f + 0.5 for f in F}
    sta = {f: f + 1 for f in F}

    DA = [(f, f + 1) for f in F]
    AA = [(f - 0.5, f + 0.5) for f in range(1, num_flights + 1)]

    AAF = {
        f: [i for i, slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]]
        for f in F
    }
    DAF = {
        f: [i for i, slot in enumerate(DA) if std[f] <= slot[1] and std[f] >= slot[0]]
        for f in F
    }

    FAA = {asl: [f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA}
    FDA = {dsl: [f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA}

    F_t = {t: list(F) for t in T}
    T_f = {f: [t for t in T if f in F_t[t]] for f in F}

    flights = [f for f in F]
    FA_k = {0: [], 1: [0, 1]}
    AK_f = {0: 1, 1: 1}
    FD_k = {0: [0, 1], 1: []}

    DA_k = {}
    for f in F:
        DA_k[f] = -1

    for f in F:
        for k in K:
            if f in FD_k[k]:
                DA_k[f] = k

    DK_f = {}
    for f in F:
        DK_f[f] = -1

    for f in F:
        for k in K:
            if f in FD_k[k]:
                DK_f[f] = k

    CF_f = {
        f: [fd for fd in F if AK_f[f] == DK_f[fd] and sta[f] <= std[fd] and fd != f]
        for f in F
    }

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

    oc = {(t, f): 1500 for f in F for t in T}
    dc = {f: 100 for f in F}
    n = {(v, P.index(p)): 25 for p in P for v in Y}
    q = {t: 100 for t in T}
    rc = {
        (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 0.5)(p, pd)
        for p in P
        for pd in P
    }
    theta = {
        (v, P.index(p), P.index(pd), z): 0 for v in Y for p in P for pd in P for z in Z
    }
    tb = {(t, k): 1 if k == 0 else 0 for t in T for k in K}
    scA = {asl: 1 for asl in AA}
    scD = {dsl: 1 for dsl in DA}
    sb = {f: 0 for f in F}
    mtt = {(t, f, fd): 0 for t in T for f in F for fd in F}
    mct = {(P.index(p), f, fd): 0 for p in P for f in F for fd in F}
    ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}
    CF_p = {P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])] for p in P}
    lf = {(P.index(p), f): 1 if p[-1] == f else 0 for p in P for f in F}
    small_theta = {z: 1000 for z in Z}
    fc = {f: 100 for f in F}
    pc = {(z, P.index(p), P.index(pd)): 0 for z in Z for p in P for pd in P}
    kappa = 100
    x_hat = {(f, t): 1 if t == f or t + 1 == f else 0 for f in F for t in T}

    variables = generate_variables(
        passenger_itinerary_disruption, T, F, Y, Z, P, AA, DA, CO_p
    )
    set_objective(
        passenger_itinerary_disruption,
        variables,
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
    )
    flight_scheduling_constraints(passenger_itinerary_disruption, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        passenger_itinerary_disruption, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        passenger_itinerary_disruption, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        passenger_itinerary_disruption,
        variables,
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
    )
    flight_delay_constraints(
        passenger_itinerary_disruption, variables, T, F, T_f, CF_f, sb, mtt, ct
    )
    itinerary_feasibility_constraints(
        passenger_itinerary_disruption, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        passenger_itinerary_disruption, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    beta_linearizing_constraints(
        passenger_itinerary_disruption, variables, Y, Z, P, CO_p
    )

    print("optimizing...")
    passenger_itinerary_disruption.optimize()

    print("generating output...")
    generate_output(
        passenger_itinerary_disruption,
        variables,
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
    )

    _, z, _, _, _, _, h, lambd, _, _, _, _, _, _, _, _ = variables

    # Cancellations and reschedules occur as expected
    for f in F:
        if f == 0:
            assert z[f].x > 0.9
        else:
            assert z[f].x < 0.9
    for p in P:
        if P.index(p) == 0:
            assert lambd[P.index(p)].x > 0.9
        else:
            assert lambd[P.index(p)].x < 0.9
    for p in P:
        for pd in P:
            if p != pd:
                for v in Y:
                    if P.index(p) == 0 and P.index(pd) == 1:
                        assert int(h[P.index(p), P.index(pd), v].x) == 25
                    else:
                        assert int(h[P.index(p), P.index(pd), v].x) == 0


def test_basic_maintenance():
    """
    Ensures that a flight is rescheduled to a cheaper tail if it is available.
    """

    basic_maintenance = Model("test_basic_maintenance")

    num_flights = 2
    num_tails = 1
    num_airports = 3
    num_fare_classes = 2
    num_delay_levels = 2
    num_time_instances = 2

    T = range(num_tails)
    F = range(num_flights)
    P = [[i, i + 1] for i in range(0, num_flights, 2)]
    K = range(num_airports)
    Y = range(num_fare_classes)
    Z = range(num_delay_levels)

    std = {0: 0.5, 1: 30.5}
    sta = {0: 5.5, 1: 35.5}

    # MAINTENANCE DATA / SETS
    PI = range(num_time_instances)
    MO = set([(f, fd) for f in F for fd in F if std[f] < std[fd]])
    F_pi = {
        pi: [f for f in F if sta[f] <= (1 + pi) * (TIME_HORIZON / num_time_instances)]
        for pi in PI
    }
    K_m = {1}
    T_m = set(T)
    PI_m = set(PI)

    abh = {0: 0.5}
    sbh = {0: 25, 1: (72 - 35.5)}
    mbh = {0: 14 * 24}
    mt = {0: 15}
    aw = {0: 0, 1: 1000, 2: 0}

    DA = [(0, 1), (30, 31)]
    AA = [(5, 6), (35, 36)]

    AAF = {
        f: [i for i, slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]]
        for f in F
    }
    DAF = {
        f: [i for i, slot in enumerate(DA) if std[f] <= slot[1] and std[f] >= slot[0]]
        for f in F
    }

    FAA = {asl: [f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA}
    FDA = {dsl: [f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA}

    F_t = {t: list(F) for t in T}
    T_f = {f: [t for t in T if f in F_t[t]] for f in F}

    flights = [f for f in F]
    FA_k = {}
    for k in K:
        FA_k[k] = []

    AK_f = {}

    for f in F:
        if f % 2 == 0:
            FA_k[1].append(f)
            AK_f[f] = 1
        else:
            FA_k[2].append(f)
            AK_f[f] = 2

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

    CF_f = {
        f: [fd for fd in F if AK_f[f] == DK_f[fd] and sta[f] <= std[fd] and fd != f]
        for f in F
    }

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
    oc = {(t, f): 100 for f in F for t in T}
    dc = {f: 100 for f in F}
    n = {(v, P.index(p)): 50 for p in P for v in Y}
    q = {t: 100 for t in T}
    rc = {
        (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 0.5)(p, pd)
        for p in P
        for pd in P
    }
    theta = {
        (v, P.index(p), P.index(pd), z): 0 for v in Y for p in P for pd in P for z in Z
    }

    tb = {(t, k): 1 if k == 0 else 0 for t in T for k in K}
    scA = {asl: 1 for asl in AA}
    scD = {dsl: 1 for dsl in DA}
    sb = {f: 0 for f in F}
    mtt = {(t, f, fd): 0 for t in T for f in F for fd in F}
    mct = {(P.index(p), f, fd): 0 for p in P for f in F for fd in F}
    ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}
    CF_p = {P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])] for p in P}
    lf = {(P.index(p), f): 1 if p[-1] == f else 0 for p in P for f in F}
    small_theta = {z: 1000 for z in Z}
    fc = {f: 100 for f in F}
    pc = {(z, P.index(p), P.index(pd)): 0 for z in Z for p in P for pd in P}
    kappa = 100
    x_hat = {(f, t): 1 if t == f or t + 1 == f else 0 for f in F for t in T}

    variables = generate_variables(basic_maintenance, T, F, Y, Z, P, AA, DA, CO_p, K)
    set_objective(
        basic_maintenance,
        variables,
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
    )
    flight_scheduling_constraints(basic_maintenance, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        basic_maintenance, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        basic_maintenance, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        basic_maintenance,
        variables,
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
    )
    flight_delay_constraints(basic_maintenance, variables, T, F, T_f, CF_f, sb, mtt, ct)
    itinerary_feasibility_constraints(
        basic_maintenance, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        basic_maintenance, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    maintenance_schedule_constraints(
        basic_maintenance, variables, T_m, sta, T_f, F, F_t, mt, MO, std
    )
    workshop_schedule_constraints(
        basic_maintenance, variables, F_t, T_m, K_m, F, T_f, K, aw, FA_k
    )
    maintenance_check_constraints(
        basic_maintenance, variables, T_m, PI_m, F_pi, sbh, mbh, F_t, MO, abh, F, T_f
    )
    beta_linearizing_constraints(basic_maintenance, variables, Y, Z, P, CO_p)

    print("optimizing...")
    basic_maintenance.optimize()

    print("generating output...")
    generate_output(
        basic_maintenance,
        variables,
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
        T_m,
    )

    (
        _,
        z,
        _,
        _,
        _,
        _,
        _,
        lambd,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = variables

    # No flights cancelled and no itineraries disrupted
    for f in F:
        assert z[f].x < 0.9
    for p in P:
        assert lambd[P.index(p)].x < 0.9
