import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Generator

import numpy as np
import pyomo.kernel as pmo
from pyomo.environ import (
    AbstractModel,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    Param,
    RangeSet,
    Reals,
    SolverFactory,
    Var,
    floor,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition

logger = logging.getLogger(__name__)

# papers on food degradation and ice crystals
# https://www.sciencedirect.com/science/article/pii/S0260877410002013?casa_token=YlidsOemDNwAAAAA:dM94qp8mLCQ_C-9bBNKDl72UJNIU4RToFmq4KzYST4uxdbkjsIT-oQkHzknyiZXz6st3CzuR9gi2
# https://link.springer.com/article/10.1007/s12393-020-09255-8#Tab3
# https://www.sciencedirect.com/science/article/abs/pii/S0309174013001563
# Best one: describes that size and duration of fluctuations is bad
# https://www.sciencedirect.com/science/article/pii/S0023643814003715?via%3Dihub


@dataclass
class OptimizationInstance:
    lambda_mfrr: np.ndarray
    lambda_rp: np.ndarray
    lambda_spot: np.ndarray
    up_regulation_event: np.ndarray
    probabilities: np.ndarray

    elafgift: float
    moms: float
    tariff: np.ndarray

    t_c_data: np.ndarray

    nb_scenarios: int = field(init=False)

    c_f: float
    r_cf: float
    c_c: float
    r_ci: np.ndarray
    t_i: np.ndarray
    od: np.ndarray
    eta: float

    temperature_filter: np.ndarray
    mask: np.ndarray

    p_nom: float
    p_min: float
    p_base: np.ndarray
    delta_max: float

    one_lambda: bool

    dt: float
    n_steps: int
    n_hours: int = field(init=False)
    hour_steps: int = field(init=False)

    max_up_time: int
    min_up_time: int
    setpoint: float
    setpoint_ts: np.ndarray
    rebound: int

    alpha: float
    k: float

    epsilon: float

    M: int
    max_lambda_bid: float

    def __post_init__(self) -> None:
        self.nb_scenarios = self.up_regulation_event.shape[0]
        assert self.n_steps * self.dt % 1 == 0
        self.n_hours = int(self.n_steps * self.dt)
        assert 1 / self.dt % 1 == 0
        self.hour_steps = int(1 / self.dt)
        assert self.tariff.shape[0] == self.n_hours

    def reduce_instance(self, nb: int = 1) -> None:
        # Reduce instance to "nb" scenarios.
        # Function can, e.g., used for oos evaluation
        self.nb_scenarios = nb
        self.lambda_mfrr = self.lambda_mfrr[0:nb, :]
        self.lambda_rp = self.lambda_rp[0:nb, :]
        self.lambda_spot = self.lambda_spot[0:nb, :]
        self.up_regulation_event = self.up_regulation_event[0:nb, :]
        self.probabilities = np.array([1 for _ in range(nb)])

    def convert_to_robust_scenario(self) -> None:
        self.reduce_instance()
        self.lambda_rp = self.lambda_spot * 2
        self.up_regulation_event = np.ones(shape=(1, 24))

    def lookback_generator(self, lookback: int) -> Generator:
        # Generator that returns chunks of the instance
        assert lookback > 0
        assert lookback < self.nb_scenarios
        for i in range(lookback, self.nb_scenarios):
            yield self(i - lookback, i), self(i, i + 1)

    def chunk_generator(self, chunk_size: int) -> Generator:
        # Generator that returns chunks of the instance
        assert chunk_size > 0
        if chunk_size >= self.nb_scenarios:
            for _ in range(1):
                yield self(0, self.nb_scenarios)
        else:
            for i in range(0, self.nb_scenarios, chunk_size):
                yield self(i, i + chunk_size)

    def __call__(self, lb: int, ub: int) -> "OptimizationInstance":
        # Return a chunk of the instance with the given index
        assert lb >= 0
        assert lb <= self.nb_scenarios - 1
        assert ub > 0
        assert ub > lb
        ub = min(self.nb_scenarios, ub)
        nb_scenarios = ub - lb

        _copy = deepcopy(self)
        _copy.nb_scenarios = nb_scenarios
        _copy.lambda_mfrr = self.lambda_mfrr[lb:ub, :].reshape(nb_scenarios, -1)
        _copy.lambda_rp = self.lambda_rp[lb:ub, :].reshape(nb_scenarios, -1)
        _copy.lambda_spot = self.lambda_spot[lb:ub, :].reshape(nb_scenarios, -1)
        _copy.up_regulation_event = self.up_regulation_event[lb:ub, :].reshape(
            nb_scenarios, -1
        )
        _copy.probabilities = np.ones(nb_scenarios) / nb_scenarios
        assert _copy.up_regulation_event.shape == (nb_scenarios, self.n_hours)
        assert _copy.up_regulation_event.shape == _copy.lambda_mfrr.shape
        assert _copy.up_regulation_event.shape == _copy.lambda_rp.shape
        assert _copy.up_regulation_event.shape == _copy.lambda_spot.shape

        return _copy


def pyomo_get_abstract_model() -> AbstractModel:

    model = AbstractModel()

    model._n_steps = Param(initialize=96, mutable=False)
    model.hour_steps = Param(initialize=4, mutable=False)
    model._n_hours = Param(initialize=24, mutable=False)
    model._nb_scenarios = Param(initialize=1, mutable=True)

    model.time_steps = RangeSet(0, model._n_steps - 1)
    model.n_hours = RangeSet(0, model._n_hours - 1)
    model.nb_scenarios = RangeSet(0, model._nb_scenarios - 1)

    model.M = Param(initialize=15, mutable=True)
    model.max_lambda_bid = Param(initialize=50, mutable=True)
    # one_lambda = inst.one_lambda

    model.p_base = Param(model.n_hours, mutable=True, domain=Reals)
    model.p_nom = Param(mutable=True)
    model.p_min = Param(mutable=True)

    model.c_f = Param(mutable=True)
    model.r_cf = Param(mutable=True)
    model.c_c = Param(mutable=True)
    model.r_ci = Param(model.time_steps, mutable=True)
    model.t_i = Param(model.time_steps, mutable=True)
    model.od = Param(model.time_steps, mutable=True)
    model.setpoint_ts = Param(model.time_steps, mutable=True, domain=Reals)
    model.eta = Param(mutable=True)
    model.epsilon = Param(mutable=True)

    model.temperature_filter = Param(model.time_steps, mutable=True, domain=Reals)
    model.mask = Param(model.n_hours, mutable=True, domain=Reals)

    # true temperature. Not used
    model.t_c_data = Param(model.time_steps, mutable=True, domain=Reals)

    # variables for thermodynamics
    model.t_c = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.t_c_base = Var(model.time_steps, domain=Reals)
    model.t_f = Var(model.nb_scenarios, model.time_steps, domain=Reals)
    model.t_f_base = Var(model.time_steps, domain=Reals)

    model.dt = Param(mutable=True)
    model.delta_max = Param(mutable=True)

    model.rebound = Param(mutable=True)
    model.setpoint = Param(mutable=True)
    model.max_up_time = Param(mutable=True)
    model.min_up_time = Param(mutable=True)

    model.lambda_spot = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.lambda_rp = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.lambda_mfrr = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.up_regulation_event = Param(
        model.nb_scenarios, model.n_hours, mutable=True, domain=Reals
    )
    model.probabilities = Param(model.nb_scenarios, mutable=True)

    model.elafgift = Param(mutable=True)
    model.moms = Param(mutable=True)
    model.tariff = Param(model.n_hours, mutable=True, domain=Reals)

    model.u_up = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.u_down = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.y_up = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.y_down = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.z_up = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.z_down = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)

    # temperature deviation
    model.delta = Var(domain=NonNegativeReals)

    # regulation
    model.p_up = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.p_down = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.pt = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.s = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)

    # reservation capacity (up mFRR)
    # _p_up = {j: 0 for j in model.n_hours}
    model.p_up_reserve = Var(model.n_hours, domain=NonNegativeReals)

    # CVaR variables
    # model.cvar = Var(domain=Reals)
    # model.var = Var(domain=Reals)
    # model.eta_var = Var(model.nb_scenarios, domain=NonNegativeReals)
    # model.alpha = Param(mutable=True)
    # model.k = Param(mutable=True)

    # variables related to bid strategy
    # _lambda_b = {(i, j): 0.1 for i in model.nb_scenarios for j in model.n_hours}
    model.lambda_b = Var(
        model.nb_scenarios,
        model.n_hours,
        domain=NonNegativeReals,
        # initialize=_lambda_b,
    )
    model.phi = Var(model.nb_scenarios, model.n_hours, domain=NonNegativeReals)
    model.g = Var(model.nb_scenarios, model.n_hours, domain=pmo.Binary)
    model.alpha = Var(domain=NonNegativeReals, initialize=1)
    model.beta = Var(domain=NonNegativeReals, initialize=1)

    model.constraints = ConstraintList()

    return model


class SolverInstance:
    def __init__(self) -> None:
        self.model = pyomo_get_abstract_model()

        self.add_power_constraints()
        self.add_bid_constraints()
        self.add_thermodynamics_constraints()
        self.add_auxillary_constraints()
        self.add_rebound_constraints()

    @staticmethod
    def instance_to_dict(inst: OptimizationInstance) -> Dict:
        data: Dict[None, Any] = {None: {}}

        data[None]["_n_steps"] = {None: inst.n_steps}
        data[None]["hour_steps"] = {None: inst.hour_steps}
        data[None]["_n_hours"] = {None: inst.n_hours}
        data[None]["_nb_scenarios"] = {None: inst.nb_scenarios}

        data[None]["M"] = {None: inst.M}
        data[None]["max_lambda_bid"] = {None: inst.max_lambda_bid}

        data[None]["p_base"] = {i: e for i, e in enumerate(inst.p_base)}
        data[None]["p_nom"] = {None: inst.p_nom}
        data[None]["p_min"] = {None: inst.p_min}

        data[None]["c_f"] = {None: inst.c_f}
        data[None]["r_cf"] = {None: inst.r_cf}
        data[None]["c_c"] = {None: inst.c_c}
        data[None]["r_ci"] = {i: e for i, e in enumerate(inst.r_ci)}
        data[None]["eta"] = {None: inst.eta}
        data[None]["epsilon"] = {None: inst.epsilon}
        data[None]["t_i"] = {i: e for i, e in enumerate(inst.t_i)}
        data[None]["t_c_data"] = {i: e for i, e in enumerate(inst.t_c_data)}
        data[None]["od"] = {i: e for i, e in enumerate(inst.od)}
        data[None]["temperature_filter"] = {
            i: e for i, e in enumerate(inst.temperature_filter)
        }
        data[None]["mask"] = {i: e for i, e in enumerate(inst.mask)}

        data[None]["dt"] = {None: inst.dt}
        data[None]["delta_max"] = {None: inst.delta_max}

        data[None]["rebound"] = {None: inst.rebound}
        data[None]["setpoint"] = {None: inst.setpoint}
        data[None]["setpoint_ts"] = {i: e for i, e in enumerate(inst.setpoint_ts)}
        data[None]["max_up_time"] = {None: inst.max_up_time}
        data[None]["min_up_time"] = {None: inst.min_up_time}

        data[None]["lambda_spot"] = {
            (i, j): inst.lambda_spot[i, j]
            for i in range(inst.lambda_spot.shape[0])
            for j in range(inst.lambda_spot.shape[1])
        }
        data[None]["lambda_rp"] = {
            (i, j): inst.lambda_rp[i, j]
            for i in range(inst.lambda_rp.shape[0])
            for j in range(inst.lambda_rp.shape[1])
        }
        data[None]["lambda_mfrr"] = {
            (i, j): inst.lambda_mfrr[i, j]
            for i in range(inst.lambda_mfrr.shape[0])
            for j in range(inst.lambda_mfrr.shape[1])
        }
        data[None]["up_regulation_event"] = {
            (i, j): inst.up_regulation_event[i, j]
            for i in range(inst.up_regulation_event.shape[0])
            for j in range(inst.up_regulation_event.shape[1])
        }

        data[None]["elafgift"] = {None: inst.elafgift}
        data[None]["moms"] = {None: inst.moms}
        data[None]["tariff"] = {i: e for i, e in enumerate(inst.tariff)}

        data[None]["probabilities"] = {i: e for i, e in enumerate(inst.probabilities)}
        # data[None]["alpha"] = {None: inst.alpha}
        # data[None]["k"] = {None: inst.k}

        return data

    def add_power_constraints(self) -> None:
        def power_constraint_1(m, w, t):  # type:ignore
            return m.pt[w, t] == m.p_base[t] - m.p_up[w, t] + m.p_down[w, t]

        def power_constraint_2(m, w, t):  # type:ignore
            return m.p_up[w, t] <= m.u_up[w, t] * (m.p_base[t] - m.p_min)

        def power_constraint_2_2(m, w, t):  # type:ignore
            return m.p_up[w, t] <= m.p_up_reserve[t] * m.up_regulation_event[w, t]

        def power_constraint_3(m, w, t):  # type:ignore
            return m.p_down[w, t] <= m.u_down[w, t] * m.p_nom

        def power_constraint_4(m, w, t):  # type:ignore
            return m.p_up[w, t] + m.s[w, t] >= m.u_up[w, t] * m.p_base[t]

        def power_constraint_4_2(m, w, t):  # type:ignore
            return (
                m.p_up[w, t]
                >= m.p_up_reserve[t] * m.up_regulation_event[w, t] - m.s[w, t]
            )

        def power_constraint_5(m, w, t):  # type:ignore
            return m.p_down[w, t] >= m.u_down[w, t] * 0.10 * (m.p_nom - m.p_base[t])

        def power_constraint_6(m, w, t):  # type:ignore
            return m.pt[w, t] <= m.p_nom

        def power_constraint_6_2(m, w, t):  # type:ignore
            return m.pt[w, t] >= m.p_min

        def power_constraint_7(m, w, t):  # type:ignore
            return m.s[w, t] >= 0

        def power_constraint_8(m, w, t):  # type:ignore
            return m.s[w, t] <= m.p_base[t]

        def power_constraint_9(m, t):  # type:ignore
            return m.p_up_reserve[t] <= m.p_base[t] * (1 - m.mask[t])

        def power_constraint_9_2(m, t):  # type:ignore
            return m.p_up_reserve[t] >= m.p_base[t] * (1 - m.mask[t])

        def power_constraint_10(m, w, t):  # type:ignore
            return m.p_up[w, t] <= m.p_nom * (1 - m.mask[t])

        def power_constraint_11(m, w, t):  # type:ignore
            return m.p_down[w, t] <= m.p_nom * (1 - m.mask[t])

        def power_constraint_test(m, w, t):  # type:ignore
            return m.pt[w, t] == m.p_base[t]

        # add power constraints to model
        self.model.P1 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_1,
        )
        self.model.P2 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_2,
        )
        self.model.P22 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_2_2,
        )
        self.model.P3 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_3,
        )
        # self.model.P4 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=power_constraint_4,
        # )
        # self.model.P44 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=power_constraint_4_2,
        # )
        self.model.P5 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_5,
        )
        self.model.P6 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_6,
        )
        self.model.P6_2 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_6_2,
        )
        self.model.P7 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_7,
        )
        self.model.P8 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_8,
        )
        self.model.P9 = Constraint(
            self.model.n_hours,
            rule=power_constraint_9,
        )
        # self.model.P9_2 = Constraint(
        #     self.model.n_hours,
        #     rule=power_constraint_9_2,
        # )
        self.model.P10 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_10,
        )
        self.model.P11 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=power_constraint_11,
        )
        # self.model.P_TEST = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=power_constraint_test,
        # )

    def add_bid_constraints(self) -> None:
        def lambda_bid_1(m, w, t):  # type:ignore
            # g == 1 when RP price difference is above our bid price
            return (m.lambda_rp[w, t] - m.lambda_spot[w, t]) >= m.lambda_b[
                w, t
            ] - m.M * (1 - m.g[w, t])

        def lambda_bid_2(m, w, t):  # type:ignore
            return (
                m.lambda_b[w, t]
                >= (m.lambda_rp[w, t] - m.lambda_spot[w, t]) - m.M * m.g[w, t]
            )

        def phi_less_than(m, w, t):  # type:ignore
            return m.phi[w, t] <= m.p_base[t] * (1 - m.mask[t])

        def lambda_bid_less_than(m, w, t):  # type:ignore
            # return m.lambda_b[t] <= 0
            return m.lambda_b[w, t] <= m.max_lambda_bid

        def real_time_power_less_than(m, w, t):  # type:ignore
            return m.p_up[w, t] <= m.phi[w, t] * m.up_regulation_event[w, t]

        def real_time_power_greater_than(m, w, t):  # type:ignore
            return m.p_up[w, t] + m.s[w, t] >= m.phi[w, t] * m.up_regulation_event[w, t]

        def phi_linearize_1(m, w, t):  # type:ignore
            return -m.g[w, t] * m.M <= m.phi[w, t]

        def phi_linearize_2(m, w, t):  # type:ignore
            return m.phi[w, t] <= m.g[w, t] * m.M

        def phi_linearize_3(m, w, t):  # type:ignore
            return -(1 - m.g[w, t]) * m.M <= m.phi[w, t] - m.p_up_reserve[t]

        def phi_linearize_4(m, w, t):  # type:ignore
            return m.phi[w, t] - m.p_up_reserve[t] <= (1 - m.g[w, t]) * m.M

        def one_lambda_only(m, w, t):  # type:ignore
            if t == 0:
                return Constraint.Skip
            return m.lambda_b[w, t] == m.lambda_b[w, t - 1]

        def one_lambda_only2(m, w, t):  # type:ignore
            if w == 0:
                return Constraint.Skip
            return m.lambda_b[w, t] == m.lambda_b[w - 1, t]

        def lambda_test_1(m, w, t):  # type:ignore
            # if w < 2:
            #     return Constraint.Skip
            # return m.lambda_b[w - 2, t] == m.lambda_b[w, t]
            # return (
            #     m.lambda_b[w, t]
            #     == (
            #         max(value(self.model.lambda_spot[w, :]))
            #         - m.lambda_spot[w, t]
            #     )
            #     * m.alpha
            #     + m.beta
            # )
            return (
                m.lambda_b[w, t]
                == (np.diff(value(m.lambda_spot[w, :]), append=5))[t] * m.alpha + m.beta
            )
            # return (
            #     m.lambda_b[w, t]
            #     == np.minimum(
            #         np.exp((np.diff(value(m.lambda_spot[w, :]), append=1))), 5
            #     )[t]
            # )

        self.model.Bid1 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=lambda_bid_1,
        )
        self.model.Bid2 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=lambda_bid_2,
        )
        # self.model.Bid3 = Constraint(
        # self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=phi_less_than,
        # )
        self.model.Bid4 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=lambda_bid_less_than,
        )
        self.model.Bid5 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=real_time_power_less_than,
        )
        self.model.Bid6 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=real_time_power_greater_than,
        )
        self.model.Bid7 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=phi_linearize_1,
        )
        self.model.Bid8 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=phi_linearize_2,
        )
        self.model.Bid9 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=phi_linearize_3,
        )
        self.model.Bid10 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=phi_linearize_4,
        )
        self.model.one_lambda_constraint_1 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=one_lambda_only,
        )
        self.model.one_lambda_constraint_2 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=one_lambda_only2,
        )
        self.model.lambda_policy_1 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=lambda_test_1,
        )

    def add_thermodynamics_constraints(self) -> None:
        def t_c_constraint_1(m, w, t):  # type:ignore
            return (
                m.t_c_base[t] - m.delta <= m.t_c[w, t] + m.temperature_filter[t] * 100
            )

        def t_c_constraint_2(m, w, t):  # type:ignore
            return (
                m.t_c_base[t] + m.delta >= m.t_c[w, t] - m.temperature_filter[t] * 100
            )

        def t_c_constraint_3(m):  # type:ignore
            return m.delta <= m.delta_max

        def t_f_baseline(m, t):  # type:ignore
            if t > 0:
                return (
                    m.t_f_base[t]
                    == m.t_f_base[t - 1]
                    + 1 / m.c_f * (m.t_c_base[t - 1] - m.t_f_base[t - 1]) * m.dt
                )
            return m.t_f_base[0] == m.setpoint_ts[0]

        def t_c_baseline(m, t):  # type:ignore
            if t > 0:
                h = floor(t / m.hour_steps)  # only one power step per hour
                return m.t_c_base[t] == m.t_c_base[t - 1] + 1 / m.c_c * (
                    1 / m.r_cf * (m.t_f_base[t - 1] - m.t_c_base[t - 1]) * m.dt
                    + 1 / m.r_ci[t] * (m.t_i[t] - m.t_c_base[t - 1]) * m.dt
                    - m.eta * m.p_base[h] * m.od[t] * m.dt
                    + m.epsilon * m.temperature_filter[t] * m.dt
                )

            return m.t_c_base[0] == m.setpoint_ts[0]

        def tf_constraint(m, w, t):  # type:ignore
            if t > 0:
                return (
                    m.t_f[w, t]
                    == m.t_f[w, t - 1]
                    + 1 / m.c_f * (m.t_c[w, t - 1] - m.t_f[w, t - 1]) * m.dt
                )
            return m.t_f[w, 0] == m.setpoint_ts[0]

        def tc_constraint(m, w, t):  # type:ignore
            if t > 0:
                h = floor(t / m.hour_steps)  # only one power step per hour
                return m.t_c[w, t] == m.t_c[w, t - 1] + 1 / m.c_c * (
                    1 / m.r_cf * (m.t_f[w, t - 1] - m.t_c[w, t - 1]) * m.dt
                    + 1 / m.r_ci[t] * (m.t_i[t] - m.t_c[w, t - 1]) * m.dt
                    - m.eta * m.pt[w, h] * m.od[t] * m.dt
                    + m.epsilon * m.temperature_filter[t] * m.dt
                )

            return m.t_c[w, 0] == m.setpoint_ts[0]

        def boundary_constraint(m, w):  # type:ignore
            # teperature at the end must equal tmeperature for the beginning (or data)
            return m.t_f[w, m._n_steps - 1] <= m.t_f_base[m._n_steps - 1] + 0.1

        # add system constraints to instance
        self.model.S1 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=t_c_constraint_1,
        )
        self.model.S2 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=t_c_constraint_2,
        )
        self.model.S3 = Constraint(rule=t_c_constraint_3)
        self.model.S4 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=tf_constraint,
        )
        self.model.S5 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=tc_constraint,
        )
        self.model.BaselineFood = Constraint(
            self.model.time_steps,
            rule=t_f_baseline,
        )
        self.model.BaselineAir = Constraint(
            self.model.time_steps,
            rule=t_c_baseline,
        )
        self.model.S6 = Constraint(
            self.model.nb_scenarios,
            rule=boundary_constraint,
        )

    def add_auxillary_constraints(self) -> None:
        # auxillary constraints
        def u_up_rule(m, w, t):  # type:ignore
            if t > 0:
                return (
                    m.u_up[w, t - 1] - m.u_up[w, t] + m.y_up[w, t] - m.z_up[w, t] == 0
                )
            return m.u_up[w, t] == 0

        def y_z_up_rule(m, w, t):  # type:ignore
            return m.y_up[w, t] + m.z_up[w, t] <= 1

        def u_down_rule(m, w, t):  # type:ignore
            if t > 0:
                return (
                    m.u_down[w, t - 1]
                    - m.u_down[w, t]
                    + m.y_down[w, t]
                    - m.z_down[w, t]
                    == 0
                )
            return m.u_down[w, t] == 0

        def y_z_down_rule(m, w, t):  # type:ignore
            return m.y_down[w, t] + m.z_down[w, t] <= 1

        def u_rule(m, w, t):  # type:ignore
            return m.u_up[w, t] + m.u_down[w, t] <= 1

        def y_rule(m, w, t):  # type:ignore
            return m.y_up[w, t] + m.y_down[w, t] <= 1

        def z_rule(m, w, t):  # type:ignore
            return m.z_up[w, t] + m.z_down[w, t] <= 1

        def min_up_reguluation_rule(m, w, t):  # type:ignore
            ub = min(t + value(m.min_up_time) + 1, value(m._n_hours))
            return (
                sum(m.u_up[w, k] for k in range(t, ub)) >= m.min_up_time * m.y_up[w, t]
            )

        def max_up_reguluation_rule(m, w, t):  # type:ignore
            ub = min(t + value(m.max_up_time) + 1, value(m._n_hours))
            return sum(m.u_up[w, k] for k in range(t, ub)) <= m.max_up_time

        # add all auxillary constraints to instance
        self.model.A1 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=u_up_rule,
        )
        self.model.A2 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=y_z_up_rule,
        )
        self.model.A3 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=u_down_rule,
        )
        self.model.A4 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=y_z_down_rule,
        )
        self.model.A5 = Constraint(
            self.model.nb_scenarios, self.model.n_hours, rule=u_rule
        )
        self.model.A6 = Constraint(
            self.model.nb_scenarios, self.model.n_hours, rule=y_rule
        )
        self.model.A7 = Constraint(
            self.model.nb_scenarios, self.model.n_hours, rule=z_rule
        )
        self.model.A8 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=min_up_reguluation_rule,
        )
        self.model.A9 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=max_up_reguluation_rule,
        )

    def add_rebound_constraints(self) -> None:
        # add rebound contraints
        def rebound_rule_1(m, w, t):  # type:ignore
            return m.y_down[w, t] >= m.z_up[w, t]

        def rebound_rule_1_2(m, w, t):  # type:ignore
            return m.y_down[w, t] <= m.z_up[w, t]

        def rebound_rule_2(m, w, t):  # type:ignore
            ub = min(t + value(m.rebound) + 1, value(m._n_hours))
            return sum(m.u_down[w, k] for k in range(t, ub)) <= value(m.rebound)

        def rebound_rule_2_2(m, w, t):  # type:ignore
            ub = min(t + value(m.rebound) + 1, value(m._n_hours))
            return (
                sum(m.u_down[w, k] for k in range(t, ub))
                # >= value(m.rebound) * m.z_up[w, t]
                >= 1 * m.z_up[w, t]
            )

        # not used
        def rebound_rule_3(m, w, t):  # type:ignore
            ub = min(
                t + value(m.rebound) + value(m.max_up_time) + 1,
                value(m._n_hours),
            )
            return (
                sum(-m.p_up[w, k] + m.p_down[w, k] for k in range(t, ub))
                >= -(1 - m.y_up[w, t]) * m.p_nom
            )

        # not used
        def rebound_rule_4(m, w, t):  # type:ignore
            ub = min(
                t + value(m.rebound) + value(m.max_up_time) + 1,
                value(m._n_hours),
            )
            return (
                sum(-m.p_up[w, k] + m.p_down[w, k] for k in range(t, ub))
                <= (1 - m.y_up[w, t]) * m.p_nom
            )

        # ensure y_down is activated only once in a rebound cycle
        def rebound_rule_5(m, w, t):  # type:ignore
            ub = min(
                t + value(m.rebound) + value(m.max_up_time) + 1,
                value(m._n_hours),
            )
            return (
                sum(m.y_down[w, k] for k in range(t, ub))
                <= m.y_up[w, t] + (1 - m.y_up[w, t]) * m._n_hours
            )

        # ensure z_up is activated only once in a rebound cycle
        def rebound_rule_6(m, w, t):  # type:ignore
            ub = min(
                t + value(m.rebound) + value(m.max_up_time) + 1,
                value(m._n_hours),
            )
            return (
                sum(m.z_up[w, k] for k in range(t, ub))
                <= m.y_up[w, t] + (1 - m.y_up[w, t]) * m._n_hours
            )

        # ensure z_down is activated only once in a rebound cycle
        def rebound_rule_7(m, w, t):  # type:ignore
            ub = min(
                t + value(m.rebound) + value(m.max_up_time) + 1,
                value(m._n_hours),
            )
            return (
                sum(m.z_down[w, k] for k in range(t, ub))
                <= m.y_up[w, t] + (1 - m.y_up[w, t]) * m._n_hours
            )

        # ensure y_up is not activated in a new rebound cycle
        def rebound_rule_8(m, w, t):  # type:ignore
            ub = min(
                t + value(m.rebound) + value(m.max_up_time) + 1,
                value(m._n_hours),
            )
            return (
                sum(m.y_up[w, k] for k in range(t + 1, ub))
                <= (1 - m.y_up[w, t]) * m._n_hours
            )

        def symmetrical_rebound(m, w):  # type:ignore
            return sum(m.p_up[w, t] for t in m._n_hours) == sum(
                m.p_down[w, t] for t in m._n_hours
            )

        def symmetrical_rebound_2(m, w, t):  # type:ignore
            return (
                sum(m.p_down[w, i] - m.p_up[w, i] for i in m.n_hours)
                >= -(1 - m.z_down[w, t]) * m.p_nom * m._n_hours
            )

        def symmetrical_rebound_3(m, w, t):  # type:ignore
            return (
                sum(m.p_down[w, i] - m.p_up[w, i] for i in m._n_hours)
                <= (1 - m.z_down[w, t]) * m.p_nom * m._n_hours
            )

        def symmetrical_rebound_v2(m, w):  # type:ignore
            return sum(m.t_f[w, t] for t in m.n_steps) == sum(
                m.t_f_base[t] for t in m.n_steps
            )

        def symmetrical_rebound_2_v2(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)  # only one power step per hour
            t0 = h * m.hour_steps
            t1 = h * m.hour_steps + 4
            return (
                # sum(m.t_f[w, i] - m.t_f_base[i] for i in m.n_steps)
                sum(m.t_f[w, i] - m.t_f_base[i] for i in range(t0, t1))
                >= -(1 - m.z_down[w, h]) * 20 * m._n_steps
                # - m.g[w, h] * 20 * m._n_steps
            )

        def symmetrical_rebound_3_v2(m, w, t):  # type:ignore
            h = floor(t / m.hour_steps)  # only one power step per hour
            t0 = h * m.hour_steps
            t1 = h * m.hour_steps + 4
            return (
                # sum(m.t_f[w, i] - m.t_f_base[i] for i in range(t, m.n_steps))
                sum(m.t_f[w, i] - m.t_f_base[i] for i in range(t0, t1))
                <= (1 - m.z_down[w, h]) * 20 * m._n_steps
                # + m.g[w, h] * 20 * m.n_steps
            )

        def equal_up_and_down_regulation_hours(m, w):  # type:ignore
            return sum(m.u_up[w, t] for t in m.n_hours) == sum(
                m.u_down[w, t] for t in m.n_hours
            )

        # add all rebound constraints to instance
        self.model.C24 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=rebound_rule_1,
        )
        self.model.C24_2 = Constraint(
            self.model.nb_scenarios,
            self.model.n_hours,
            rule=rebound_rule_1_2,
        )
        # self.model.C25 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=rebound_rule_2,
        # )
        # self.model.C26 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=rebound_rule_2_2,
        # )
        # self.model.C28 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=rebound_rule_5,
        # )
        # self.model.C29 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=rebound_rule_6,
        # )
        # self.model.C30 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=rebound_rule_7,
        # )
        # self.model.C31 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=rebound_rule_8,
        # )
        # self.model.C32 = Constraint(
        #     self.model.nb_scenarios,
        #     rule=symmetrical_rebound,
        # )
        # self.model.C33 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=symmetrical_rebound_2,
        # )
        # self.model.C34 = Constraint(
        #     self.model.nb_scenarios,
        #     self.model.n_hours,
        #     rule=symmetrical_rebound_3,
        # )
        # self.model.temp_reb1 = Constraint(
        #     self.model.nb_scenarios,
        #     rule=symmetrical_rebound_v2,
        # )
        self.model.temp_reb2 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=symmetrical_rebound_2_v2,
        )
        self.model.temp_reb3 = Constraint(
            self.model.nb_scenarios,
            self.model.time_steps,
            rule=symmetrical_rebound_3_v2,
        )
        self.model.rebound_hours_1 = Constraint(
            self.model.nb_scenarios,
            rule=equal_up_and_down_regulation_hours,
        )

        # up-regulation should happen first
        def up_regulation_first_rule(m, w, t):  # type:ignore
            return sum(m.y_down[w, k] for k in range(0, t)) <= sum(
                m.y_up[w, k] for k in range(0, t)
            )

        self.model.F1 = Constraint(
            self.model.nb_scenarios,
            RangeSet(1, self.model._n_hours - 1),
            rule=up_regulation_first_rule,
        )

        # other boundary constraints
        def first_time_step_rule_1(m, w):  # type:ignore
            return m.y_down[w, 0] == 0

        def first_time_step_rule_2(m, w):  # type:ignore
            return m.z_down[w, 0] == 0

        def first_time_step_rule_3(m, w):  # type:ignore
            return m.y_up[w, 0] == 0

        def first_time_step_rule_4(m, w):  # type:ignore
            return m.z_up[w, 0] == 0

        self.model.TS1 = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_1
        )
        self.model.TS2 = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_2
        )
        self.model.TS3 = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_3
        )
        self.model.TS4 = Constraint(
            self.model.nb_scenarios, rule=first_time_step_rule_4
        )

    def add_cvar_constraints(self) -> None:
        def eta_cons(m, w):  # type:ignore
            objective_value = -sum(
                m.probabilities[w] * m.s[w, h] * 20000 / 1000 for h in m.n_hours
            )
            return m.var - objective_value <= m.eta_var[w]

        def cvar_var_cons(m):  # type:ignore
            return (
                m.var
                - 1
                / (1 - m.alpha)
                * sum(m.probabilities[w] * m.eta_var[w] for w in m.nb_scenarios)
                == m.cvar
            )

        self.model.eta_cons = Constraint(self.model.nb_scenarios, rule=eta_cons)
        self.model.cvar_var_cons = Constraint(rule=cvar_var_cons)

    @staticmethod
    def run_solver(model_instance: AbstractModel, tee: bool = True) -> Any:
        if tee:
            print(model_instance.statistics)
            print(
                f"Number of variables: {len([_ for v in model_instance.component_objects(Var, active=True) for _ in v]) }"
            )

        solver = SolverFactory("gurobi")
        if value(model_instance._nb_scenarios) > 1:  # type:ignore
            solver.options["TimeLimit"] = 60 * 4
            solver.options["MIPGap"] = 0.025
        #     pass
        # solver.options["MIPFocus"] = 1
        results = solver.solve(model_instance, tee=tee)

        ### Create the gurobi_ampl solver plugin using the ASL interface
        # solver = 'gurobi_ampl'
        # solver_io = 'nl'
        # keepfiles =     False     # True prints intermediate file names (.nl,.sol,...)
        # opt = SolverFactory(solver,solver_io=solver_io)
        # opt.options['outlev'] = 1 # tell gurobi to be verbose with output

        # if opt is None:
        #     print("")
        #     print("ERROR: Unable to create solver plugin for %s "\
        #         "using the %s interface" % (solver, solver_io))
        #     print("")

        # results = opt.solve(model_instance,
        #             keepfiles=keepfiles,
        #             tee=True)

        ### Declare all suffixes
        # The variable solution status suffix
        # (this suffix can be sent to the solver and loaded from the solution)
        # sstatus_table={'bas':1,   # basic
        #             'sup':2,   # superbasic
        #             'low':3,   # nonbasic <= (normally =) lower bound
        #             'upp':4,   # nonbasic >= (normally =) upper bound
        #             'equ':5,   # nonbasic at equal lower and upper bounds
        #             'btw':6}   # nonbasic between bounds
        # model_instance.sstatus = Suffix(direction=Suffix.IMPORT_EXPORT,
        #                     datatype=Suffix.INT)
        # model_instance.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        # # Report the best known bound on the objective function
        # model_instance.bestbound = Suffix(direction=Suffix.IMPORT)

        # # A few Gurobi variable solution sensitivity suffixes
        # model_instance.senslblo = Suffix(direction=Suffix.IMPORT) # smallest variable lower bound
        # model_instance.senslbhi = Suffix(direction=Suffix.IMPORT) # greatest variable lower bound
        # model_instance.sensublo = Suffix(direction=Suffix.IMPORT) # smallest variable upper bound
        # model_instance.sensubhi = Suffix(direction=Suffix.IMPORT) # greatest variable upper bound

        # # A Gurobi constraint solution sensitivity suffix
        # model_instance.sensrhshi = Suffix(direction=Suffix.IMPORT) # greatest right-hand side value
        # ###

        # # Tell gurobi_ampl to report solution sensitivities
        # # and bestbound via suffixes in the solution file
        # opt.options['solnsens'] = 1
        # opt.options['bestbound'] = 1

        # assert that model behave as it should

        def assert_conditions(
            phi: np.ndarray,
            p_up_reserve: np.ndarray,
            g_indicator: np.ndarray,
            lambda_b: np.ndarray,
            lambda_rp: np.ndarray,
            lambda_spot: np.ndarray,
        ) -> None:
            assert all(
                [
                    np.isclose(phi[w, t], p_up_reserve[t], atol=1e-05)
                    if g_indicator[w, t] == 1
                    else True
                    for w in range(phi.shape[0])
                    for t in range(phi.shape[1])
                ]
            )
            assert all(
                [
                    np.isclose(phi[w, t], 0, atol=1e-05)
                    if g_indicator[w, t] == 0
                    else True
                    for w in range(phi.shape[0])
                    for t in range(phi.shape[1])
                ]
            )
            # NOTE: sometime, the below test fails because g is not exactly 1.
            # We solve it by relaxing the tolerance and checking difference
            # between the balancing price and spot price
            assert all(
                [
                    np.isclose(g_indicator[w, t], 1, atol=1e-05)
                    if (
                        (lambda_rp[w, t] - lambda_spot[w, t]) > lambda_b[w, t]
                        and not (
                            np.isclose(
                                lambda_rp[w, t] - lambda_spot[w, t],
                                lambda_b[w, t],
                                atol=1e-04,
                            )
                            or lambda_rp[w, t] - lambda_spot[w, t] < 0.002
                        )
                    )
                    else True
                    for w in range(phi.shape[0])
                    for t in range(phi.shape[1])
                ]
            )
            # NOTE: sometime, the below test fails because g is not exactly 1.
            # We solve it by relaxing the tolerance and checking difference
            # between the balancing price and spot price
            assert all(
                [
                    np.isclose(g_indicator[w, t], 0, atol=1e-05)
                    if (
                        lambda_rp[w, t] - lambda_spot[w, t] < lambda_b[w, t]
                        and not (
                            np.isclose(
                                lambda_rp[w, t] - lambda_spot[w, t],
                                lambda_b[w, t],
                                atol=1e-04,
                            )
                            or lambda_rp[w, t] - lambda_spot[w, t] < 0.002
                        )
                    )
                    else True
                    for w in range(phi.shape[0])
                    for t in range(phi.shape[1])
                ]
            )

        def assert_conditions_2(
            p_up_reserve: np.ndarray,
            g_indicator: np.ndarray,
            p_up: np.ndarray,
            slack: np.ndarray,
            up_regulation_event: np.ndarray,
        ) -> None:
            assert all(
                [
                    (
                        p_up[w, t] + slack[w, t]
                        >= p_up_reserve[t]
                        * g_indicator[w, t]
                        * up_regulation_event[w, t]
                    )
                    or (
                        np.isclose(
                            p_up[w, t] + slack[w, t],
                            p_up_reserve[t]
                            * g_indicator[w, t]
                            * up_regulation_event[w, t],
                            atol=1e-03,
                        )
                    )
                    for w in range(p_up.shape[0])
                    for t in range(p_up.shape[1])
                ]
            )
            assert all(
                [
                    np.isclose(p_up[w, t], 0, atol=1e-03)
                    if (
                        up_regulation_event[w, t] == 0
                        or g_indicator[w, t] == 0
                        or p_up_reserve[t] == 0
                    )
                    else True
                    for w in range(p_up.shape[0])
                    for t in range(p_up.shape[1])
                ]
            )

        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            if tee:
                print("this is feasible and optimal")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            raise Exception("MILP is infeasible and could not be solved.")
        elif (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.feasible
        ):
            print("Solution is ok but only feasible")
        elif results.solver.termination_condition == TerminationCondition.maxTimeLimit:
            print("Time limit reached. Returning incumbent solution")
        else:
            # something else is wrong
            raise Exception(f"MILP could not be solved: {str(results.solver)}")

        # do sanity checks
        phi = np.array(list(model_instance.phi.extract_values().values())).reshape(
            -1, 24
        )
        p_up_reserve = np.array(
            list(model_instance.p_up_reserve.extract_values().values())
        ).reshape(-1)
        g_indicator = np.array(
            list(model_instance.g.extract_values().values())
        ).reshape(-1, 24)
        lambda_b = np.array(
            list(model_instance.lambda_b.extract_values().values())
        ).reshape(-1, 24)
        lambda_rp = np.array(
            list(model_instance.lambda_rp.extract_values().values())
        ).reshape(-1, 24)
        lambda_spot = np.array(
            list(model_instance.lambda_spot.extract_values().values())
        ).reshape(-1, 24)
        p_up = np.array(list(model_instance.p_up.extract_values().values())).reshape(
            -1, 24
        )
        slack = np.array(list(model_instance.s.extract_values().values())).reshape(
            -1, 24
        )
        up_regulation_event = np.array(
            list(model_instance.up_regulation_event.extract_values().values())
        ).reshape(-1, 24)

        assert_conditions(
            phi, p_up_reserve, g_indicator, lambda_b, lambda_rp, lambda_spot
        )
        assert_conditions_2(p_up_reserve, g_indicator, p_up, slack, up_regulation_event)

        if tee:
            print(f"Objective value: {model_instance.objective.expr()}")

        return model_instance, results
