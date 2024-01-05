from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from pyomo.environ import AbstractModel, Objective, maximize

from src.objective import (
    o_expected_activation_payment,
    o_expected_energy_consumption,
    o_expected_energy_cost,
    o_expected_rebound_cost,
    o_expected_reserve_payment,
    o_penalty,
    o_rule,
)
from src.problem import OptimizationInstance, SolverInstance, value

BUS = 0  # which bus to model
# temperature columns
TMP_COLS = [
    "tmp_32.0",
    "tmp_15.0",
    "tmp_16.0",
    "tmp_17.0",
    "tmp_18.0",
    "tmp_19.0",
    "tmp_20.0",
    "tmp_21.0",
    "tmp_48.0",
    "tmp_23.0",
    "tmp_25.0",
    "tmp_27.0",
    "tmp_28.0",
    "tmp_29.0",
    "tmp_30.0",
    "tmp_31.0",
]
# opening degree columns
OD_COLS = [
    "od_32.0",
    "od_15.0",
    "od_16.0",
    "od_17.0",
    "od_18.0",
    "od_19.0",
    "od_20.0",
    "od_21.0",
    "od_48.0",
    "od_23.0",
    "od_25.0",
    "od_27.0",
    "od_28.0",
    "od_29.0",
    "od_30.0",
    "od_31.0",
]
NB_BUS = 16  # number of valid buses in supermarket (for freezers)


@dataclass
class OptimizationResult:
    base_cost_today: float
    total_cost: float
    expected_energy_cost: float
    rebound_cost: float
    reserve_payment: float
    act_payment: float
    penalty_cost: float
    battery_capacity: float

    def __repr__(self) -> str:
        # print and round to 3 decimals:
        reserve_payment = f"reserve_payment {round(self.reserve_payment, 3)}"
        act_payment = f"activation_payment {round(self.act_payment, 3)}"
        expected_energy_cost = (
            f"expected_energy_cost {round(self.expected_energy_cost, 3)}"
        )
        rebound_cost = f"rebound_cost {round(self.rebound_cost, 3)}"
        total_cost = f"total_cost {round(self.total_cost, 3)}"
        base_cost_today = f"base_cost_today {round(self.base_cost_today, 3)}"
        penalty_cost = f"penalty_cost {round(self.penalty_cost, 3)}"
        battery_capacity = f"battery_capacity {round(self.battery_capacity, 3)}"
        return f"{reserve_payment}\n {act_payment}\n {expected_energy_cost}\n {rebound_cost}\n {total_cost}\n {base_cost_today}\n {penalty_cost}\n {battery_capacity}"


@dataclass
class Scenario:
    up_regulation_event: np.ndarray
    lambda_rp: np.ndarray
    lambda_spot: np.ndarray
    lambda_mfrr: np.ndarray
    prob: np.ndarray


@dataclass
class InstanceInformation:
    p_up_reserve: np.ndarray
    p_base: np.ndarray
    pt: np.ndarray
    s: np.ndarray
    slack: np.ndarray
    p_up: np.ndarray
    u_up: np.ndarray
    y_up: np.ndarray
    z_up: np.ndarray
    z_down: np.ndarray
    y_down: np.ndarray
    u_down: np.ndarray
    p_down: np.ndarray
    t_c_data: np.ndarray
    t_c: np.ndarray
    t_c_base: np.ndarray
    t_f: np.ndarray
    t_f_base: np.ndarray
    up_regulation_event: np.ndarray
    g_indicator: np.ndarray
    lambda_b: np.ndarray
    lambda_rp: np.ndarray
    lambda_spot: np.ndarray
    alpha: float
    beta: float


def build_uncertainty_set(
    scenarios: np.ndarray, nb: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(2021)
    # scenario_set = defaultdict(int)
    # for w in range(scenarios.shape[0]):
    #     s = scenarios[w,:]
    #     s = tuple(s.tolist())
    #     scenario_set[s] += 1

    # print(len(scenario_set))
    # sorted(scenario_set.items(), key=lambda x: x[1], reverse=True)
    # dict(sorted(scenario_set.items(), key=lambda x: x[1], reverse=True)).values()

    weights = (
        pd.Series(scenarios.sum(axis=1))
        .sort_values(ascending=True)
        .to_frame()
        .rename(columns={0: "count"})
    )
    weights = weights.groupby("count").size()
    probs = (weights / weights.sum()).values

    # sample scenarios according to sum of daily activation hours
    new_scenarios = np.empty(shape=(len(probs), scenarios.shape[1]))
    for j, h in enumerate(weights.index.values):
        idx = scenarios.sum(axis=1) == h
        possible_set = scenarios[idx, :]
        # get random choice
        i = rng.choice(list(range(possible_set.shape[0])), size=1)[0]
        new_scenarios[j, :] = possible_set[i, :]

    assert len(probs) == new_scenarios.shape[0]

    if nb is not None:
        assert new_scenarios.shape[0] % nb == 0 and nb < 20, (
            nb,
            new_scenarios.shape[0],
        )
        i = new_scenarios.shape[0] // nb
        new_scenarios = new_scenarios[::i, :]
        probs = probs[::i]
        probs /= probs.sum()  # type:ignore
        assert nb == new_scenarios.shape[0], (nb, new_scenarios.shape[0])

    assert len(probs) == new_scenarios.shape[0]
    assert np.isclose(1, probs.sum())

    return new_scenarios, probs


def find_rp_price(x: pd.Series) -> float:
    if x.flag == 1 and x.flag_down == 0:
        return x.BalancingPowerPriceUpDKK
    elif x.flag_down == 1 and x.flag == 0:
        return x.BalancingPowerPriceDownDKK
    elif x.flag_down == 0 and x.flag == 0:
        return x.SpotPriceDKK
    elif x.flag_down == 1 and x.flag == 1:
        if (x.SpotPriceDKK - x.BalancingPowerPriceDownDKK) > (
            x.BalancingPowerPriceUpDKK - x.SpotPriceDKK
        ):
            return x.BalancingPowerPriceDownDKK
        else:
            return x.BalancingPowerPriceUpDKK
    else:
        raise Exception


def build_uncertainty_set_v2(
    df_scenarios: pd.DataFrame, nb: int = 1, run_oos: bool = False
) -> Scenario:

    assert nb >= 1
    rng = np.random.default_rng(4)

    # Remove dates where there are less than 23 hours in a day
    dates_to_substract = (  # noqa
        df_scenarios.groupby(df_scenarios.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    df_scenarios = df_scenarios.query("Date != @dates_to_substract")

    weights = df_scenarios.groupby(df_scenarios.Date).flag.sum().value_counts()
    # probs = (weights / weights.sum()).values

    # sample scenarios according to sum of daily activation hours
    up_regulation_event = np.empty(shape=(len(weights) * nb, 24))
    prob = np.empty(len(weights) * nb)
    lambda_rp = np.empty(shape=(len(weights) * nb, 24))
    lambda_mfrr = np.empty(shape=(len(weights) * nb, 24))
    lambda_spot = np.empty(shape=(len(weights) * nb, 24))

    # overwrite lambda spot- and mFRR with average prices
    df_scenarios["lambda_rp"] = df_scenarios.apply(
        lambda x: find_rp_price(x), axis=1
    ).values
    df_scenarios["lambda_rp_diff"] = (
        df_scenarios["lambda_rp"] - df_scenarios["SpotPriceDKK"]
    )
    agg = df_scenarios.groupby("Hour").mean()
    lambda_spot[:] = np.repeat(
        agg.SpotPriceDKK.values.reshape(1, -1), len(weights) * nb, axis=0
    )
    lambda_mfrr[:] = np.repeat(
        agg.mFRR_UpPriceDKK.values.reshape(1, -1), len(weights) * nb, axis=0
    )

    # create list of all rp-up - spot prices when up-regulation happened.
    lambda_rp_up_diff = df_scenarios.query(
        "flag == 1 & flag_down == 0"
    ).lambda_rp_diff.values
    # create list of all rp - spot prices when up-regulation didn't happened.
    lambda_rp_diff = df_scenarios.query("flag == 0").lambda_rp_diff.values

    count = 0
    for j, w in weights.to_dict().items():

        dates = (
            df_scenarios.groupby(df_scenarios.Date)
            .flag.sum()
            .to_frame()
            .query(f"flag == {j}")
            .index.values.tolist()
        )
        assert len(dates) >= 1

        if j > 0:
            _dates = rng.choice(dates, replace=False, size=nb).tolist()
        else:
            # for j==0, no up-regulation happen and all days are the same
            _dates = rng.choice(dates, replace=False, size=1).tolist()

        assert len(_dates) >= 1

        for n in range(len(_dates)):

            # sample historical day where up-regulation happened
            date = _dates[n]  # noqa
            sample = df_scenarios.query("Date == @date")
            up_regulation_event[count, :] = sample.flag.values
            ix = np.where(sample.flag.values == 1)[0]

            # sample up-reg prices when up-regulation happened and rp-reg when it didn't
            lambda_rp_up_diff_sample = rng.choice(lambda_rp_up_diff, size=j)
            lambda_rp_diff_sample = rng.choice(lambda_rp_diff, size=24)

            # lambda_spot[count, :] = sample.SpotPriceDKK.values

            lambda_rp[count, :] = lambda_spot[0, :] + lambda_rp_diff_sample
            lambda_rp[count, ix] = lambda_spot[0, ix] + lambda_rp_up_diff_sample

            prob[count] = w / len(_dates)

            count += 1

    prob = prob[:count]
    prob /= sum(prob)

    if run_oos:
        pass
    else:
        # kind of robust approach..
        prob = np.ones(count) / count

    up_regulation_event = (lambda_rp > lambda_spot).astype(int)

    return Scenario(
        up_regulation_event=up_regulation_event[:count, :],
        lambda_rp=lambda_rp[:count, :] / 1000,
        lambda_spot=lambda_spot[:count, :] / 1000,
        lambda_mfrr=lambda_mfrr[:count, :] / 1000,
        prob=prob,
    )


def build_test_uncertainty_set(df_scenarios: pd.DataFrame) -> Scenario:
    rng = np.random.default_rng(4)

    # Remove dates where there are less than 23 hours in a day
    dates_to_substract = (  # noqa
        df_scenarios.groupby(df_scenarios.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    df_scenarios = df_scenarios.query("Date != @dates_to_substract")

    # choose 2 random days from the dataset
    dates = df_scenarios.Date.unique()
    dates = rng.choice(dates, size=2, replace=False).tolist()
    lambda_spot = df_scenarios.query("Date == @dates").SpotPriceDKK.values.reshape(
        2, -1
    )
    lambda_spot = np.tile(lambda_spot, (10, 1))
    lambda_rp = (
        lambda_spot
        + rng.normal(0, 300, size=lambda_spot.shape)
        + rng.integers(0, 2, size=lambda_spot.shape) * 400
    )
    lambda_rp = np.maximum(0, lambda_rp)

    # for i in range(1, lambda_rp.shape[0]):
    #     lambda_rp[i, :] = lambda_rp[0, :]

    up_regulation_event = (lambda_rp > lambda_spot).astype(int)
    lambda_mfrr = (
        df_scenarios.groupby("Hour").mFRR_UpPriceDKK.mean().values.reshape(1, -1)
    )
    lambda_mfrr = np.tile(lambda_mfrr, (20, 1))

    assert lambda_rp.shape == lambda_spot.shape
    assert lambda_rp.shape == lambda_mfrr.shape
    assert lambda_rp.shape == up_regulation_event.shape

    return Scenario(
        up_regulation_event=up_regulation_event,
        lambda_rp=lambda_rp / 1000,
        lambda_spot=lambda_spot / 1000,
        lambda_mfrr=lambda_mfrr / 1000,
        prob=np.ones(20) / 20,
    )


def build_oos_scenarios(df_scenarios: pd.DataFrame) -> Scenario:
    # Remove dates where there are less than 23 hours in a day
    dates_to_substract = (  # noqa
        df_scenarios.groupby(df_scenarios.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    df_scenarios = df_scenarios.query("Date != @dates_to_substract")
    df_scenarios["lambda_rp"] = df_scenarios.apply(
        lambda x: find_rp_price(x), axis=1
    ).values

    lambda_spot = df_scenarios.SpotPriceDKK.values.reshape(-1, 24)
    lambda_rp = df_scenarios.lambda_rp.values.reshape(-1, 24)
    lambda_mfrr = df_scenarios.mFRR_UpPriceDKK.values.reshape(-1, 24)

    up_regulation_event = (lambda_rp > lambda_spot).astype(int)

    assert lambda_rp.shape == lambda_spot.shape
    assert lambda_rp.shape == lambda_mfrr.shape
    assert lambda_rp.shape == up_regulation_event.shape

    return Scenario(
        up_regulation_event=up_regulation_event,
        lambda_rp=lambda_rp / 1000,
        lambda_spot=lambda_spot / 1000,
        lambda_mfrr=lambda_mfrr / 1000,
        prob=np.ones(lambda_rp.shape[0]) / lambda_rp.shape[0],
    )


def get_arbitrary_scenarios(
    df_scenarios: pd.DataFrame, nb_spot: int = 10, nb_rp: int = 1, seed: int = 1
) -> Scenario:
    rng = np.random.default_rng(seed)

    # Remove dates where there are less than 23 hours in a day
    dates_to_substract = (  # noqa
        df_scenarios.groupby(df_scenarios.Date)
        .flag.count()
        .to_frame()
        .query("flag <= 23")
        .index.values.tolist()
    )
    df_scenarios = df_scenarios.query("Date != @dates_to_substract")
    df_scenarios["lambda_rp"] = df_scenarios.apply(
        lambda x: find_rp_price(x), axis=1
    ).values
    df_scenarios["lambda_rp_diff"] = (
        df_scenarios["lambda_rp"] - df_scenarios["SpotPriceDKK"]
    )

    # choose 'nb_spot' random days from the dataset for spot prices
    dates = df_scenarios.Date.unique()
    _dates = rng.choice(dates, size=nb_spot, replace=False).tolist()  # noqa
    lambda_spot = df_scenarios.query("Date == @_dates").SpotPriceDKK.values.reshape(
        nb_spot, 24
    )
    lambda_spot = np.tile(lambda_spot, (nb_rp, 1))
    lambda_spot = lambda_spot.round()

    # distribution of balancing price differentials
    # TODO: use a better distribution
    assert nb_spot * nb_rp <= 365
    dates = (
        df_scenarios.groupby(df_scenarios.Date)
        .flag.sum()
        .sort_values()
        .drop_duplicates()
        .index.values.tolist()
    )
    # weights = df_scenarios.groupby(df_scenarios.Date).flag.sum().value_counts()
    replace = True if nb_spot > len(dates) else False
    _dates = rng.choice(dates, size=nb_spot, replace=replace).tolist()  # noqa
    lambda_rp_diff_set = np.array(
        [
            df_scenarios.query(f"Date == '{_date}'").lambda_rp_diff.values
            for _date in _dates
        ]
    )
    lambda_rp_diff_set = lambda_rp_diff_set.reshape(-1, 24)
    # lambda_rp_diff_set = df_scenarios.query(
    #     "Date == @_dates"
    # ).lambda_rp_diff.values.reshape(-1, 24)
    # replace = False if nb_spot * nb_rp * 24 <= lambda_rp_diff_set.shape[0] else True
    # lambda_rp_diff = rng.choice(
    #     lambda_rp_diff_set, size=nb_spot * nb_rp * 24, replace=replace
    # )  # noqa
    # lambda_rp_diff = lambda_rp_diff.reshape(nb_spot * nb_rp, 24)
    # lambda_rp_diff = np.tile(lambda_rp_diff_set, (nb_spot, 1))
    lambda_rp = lambda_spot + lambda_rp_diff_set
    lambda_rp = lambda_rp.round()

    up_regulation_event = (lambda_rp > lambda_spot).astype(int)
    lambda_mfrr = (
        df_scenarios.groupby("Hour").mFRR_UpPriceDKK.mean().values.reshape(1, -1)
    )
    lambda_mfrr = np.tile(lambda_mfrr, (nb_spot, 1))

    assert lambda_rp.shape == lambda_spot.shape
    assert lambda_rp.shape == lambda_mfrr.shape
    assert lambda_rp.shape == up_regulation_event.shape

    return Scenario(
        up_regulation_event=up_regulation_event,
        lambda_rp=lambda_rp / 1000,
        lambda_spot=lambda_spot / 1000,
        lambda_mfrr=lambda_mfrr / 1000,
        prob=np.ones(nb_spot * nb_rp) / (nb_spot * nb_rp),
    )


def get_scenarios_from_lookback(df: pd.DataFrame, lookback: int) -> Scenario:
    start_date = datetime(2022, 1, 1) - timedelta(days=lookback)
    df_scenarios = df.query(f"HourUTC >= '{start_date}'")

    lambda_spot = df_scenarios["SpotPriceDKK"].values.reshape(-1, 24)
    lambda_mfrr = df_scenarios["mFRR_UpPriceDKK"].values.reshape(-1, 24)

    lambda_rp = df_scenarios.lambda_rp.values.reshape(-1, 24)
    up_regulation_event = (lambda_rp > lambda_spot).astype(int)

    assert lambda_spot.shape == lambda_mfrr.shape
    assert lambda_rp.shape == up_regulation_event.shape
    assert lambda_rp.shape == lambda_mfrr.shape

    scenarios = Scenario(
        up_regulation_event=up_regulation_event,
        lambda_rp=lambda_rp,
        lambda_spot=lambda_spot,
        lambda_mfrr=lambda_mfrr,
        prob=np.ones(lambda_rp.shape[0]) / lambda_rp.shape[0],
    )

    return scenarios


def sample_lambda_rp(df: pd.DataFrame, size: int, seed: int) -> np.ndarray:
    # sample dates where up-regulation happend "flag.sum() times"
    dates = (
        df.groupby(df.Date)
        .flag.sum()
        .sort_values()
        .drop_duplicates()
        .index.values.tolist()
    )
    rng = np.random.default_rng(seed)
    sampled_dates = rng.choice(dates, size=size, replace=True).tolist()  # noqa
    lambda_rp_diff_set = np.array(
        [
            df.query(f"Date == '{_date}'").lambda_rp_diff.values
            for _date in sampled_dates
        ]
    ).reshape(-1, 24)

    return lambda_rp_diff_set


def get_chunk_instance_old(
    scenarios: Scenario,
    _p_base: Optional[np.ndarray] = None,
) -> OptimizationInstance:
    # prepare training instance using chunk data, i.e., an average day
    df = pd.read_csv("data/chunk2.csv")
    df_agg = df.groupby("Hour").mean()
    setpoint = -18.5

    tc_meas = df[TMP_COLS].values[:, BUS].reshape(-1, 1)
    od_meas = (df[OD_COLS].values / 100)[:, BUS].reshape(-1, 1)
    eta = 3
    R_eta_steady_state = (df.room_temp.values - setpoint) / (
        df.Po_ss.values * np.mean(od_meas) / NB_BUS * eta
    )
    day_filter = ((df.Hour.values >= 6) & (df.Hour.values <= 22)).astype(int)
    night_filter = 1 - day_filter
    r_ci = R_eta_steady_state * (day_filter * 1.246 + night_filter * 1.5)
    p_base = df_agg.Pt.values / NB_BUS
    p_nom = max(p_base) * 2
    p_min = 0
    n_steps = 24 * 4
    dt = 0.25
    ti = df.room_temp.values
    t_air = tc_meas.reshape(-1)
    temperature_filter = (np.diff(t_air, prepend=0) > 4).astype(int)
    mask = np.zeros(24)
    ix = np.floor(np.where(temperature_filter == 1)[0] / 4).astype(int)
    mask[ix] = 1
    setpoint_ts = t_air

    return OptimizationInstance(
        lambda_mfrr=scenarios.lambda_mfrr,
        lambda_rp=scenarios.lambda_rp,
        lambda_spot=scenarios.lambda_spot,
        up_regulation_event=scenarios.up_regulation_event,
        probabilities=scenarios.prob,
        c_f=5.502,
        # c_f=2,
        r_cf=4.910,
        c_c=0.129,
        r_ci=r_ci,
        t_i=ti,
        od=od_meas.reshape(-1),
        eta=2.383,
        p_nom=p_nom,
        p_min=p_min,
        p_base=_p_base if _p_base is not None else p_base,
        delta_max=50,
        dt=dt,
        n_steps=n_steps,
        max_up_time=5,
        min_up_time=1,
        setpoint=setpoint,
        setpoint_ts=setpoint_ts,
        rebound=4,
        alpha=0.95,
        k=0,
        epsilon=6.477,
        temperature_filter=temperature_filter,
        mask=mask,
        t_c_data=t_air,
        elafgift=0.0,
        moms=0.0,
        tariff=np.zeros(24),
        one_lambda=True,
        M=15,
        max_lambda_bid=5,
    )


def get_chunk_instance(
    scenarios: Scenario,
    _p_base: Optional[np.ndarray] = None,
) -> OptimizationInstance:
    # prepare training instance using chunk data, i.e., an average day
    df = pd.read_csv("data/chunk2.csv")
    df_agg = df.groupby("Hour").mean()
    setpoint = -18.5

    tc_meas = df[TMP_COLS].values[:, BUS].reshape(-1, 1)
    od_meas = (df[OD_COLS].values / 100)[:, BUS].reshape(-1, 1)
    eta = 3
    R_eta_steady_state = (df.room_temp.values - setpoint) / (
        df.Po_ss.values * np.mean(od_meas) / NB_BUS * eta
    )
    day_filter = ((df.Hour.values >= 6) & (df.Hour.values <= 22)).astype(int)
    night_filter = 1 - day_filter
    r_ci = R_eta_steady_state * (day_filter * 1.996 + night_filter * 1.977)
    p_base = df_agg.Pt.values / NB_BUS
    p_nom = max(p_base) * 2
    p_min = 0
    n_steps = 24 * 4
    dt = 0.25
    ti = df.room_temp.values
    t_air = tc_meas.reshape(-1)
    temperature_filter = (np.diff(t_air, prepend=0) > 4).astype(int)
    mask = np.zeros(24)
    ix = np.floor(np.where(temperature_filter == 1)[0] / 4).astype(int)
    mask[ix] = 1
    setpoint_ts = t_air

    return OptimizationInstance(
        lambda_mfrr=scenarios.lambda_mfrr,
        lambda_rp=scenarios.lambda_rp,
        lambda_spot=scenarios.lambda_spot,
        up_regulation_event=scenarios.up_regulation_event,
        probabilities=scenarios.prob,
        c_f=6.552,
        r_cf=5.010,
        c_c=0.077,
        r_ci=r_ci,
        t_i=ti,
        od=od_meas.reshape(-1),
        eta=1.561,
        p_nom=p_nom,
        p_min=p_min,
        p_base=_p_base if _p_base is not None else p_base,
        delta_max=50,
        dt=dt,
        n_steps=n_steps,
        max_up_time=5,
        min_up_time=1,
        setpoint=setpoint,
        setpoint_ts=setpoint_ts,
        rebound=4,
        alpha=0.95,
        k=0,
        epsilon=3.372,
        temperature_filter=temperature_filter,
        mask=mask,
        t_c_data=t_air,
        elafgift=0.0,
        moms=0.0,
        tariff=np.zeros(24),
        one_lambda=True,
        M=15,
        max_lambda_bid=5,
    )


def get_variables_and_params(instance: AbstractModel) -> InstanceInformation:
    # extract solver results
    p_up_reserve = np.array(list(instance.p_up_reserve.extract_values().values()))
    p_base = np.array(list(instance.p_base.extract_values().values()))
    pt = np.array(list(instance.pt.extract_values().values())).reshape(-1, 24)
    p_up = np.array(list(instance.p_up.extract_values().values())).reshape(-1, 24)
    u_up = np.array(list(instance.u_up.extract_values().values())).reshape(-1, 24)
    y_up = np.array(list(instance.y_up.extract_values().values())).reshape(-1, 24)
    z_up = np.array(list(instance.z_up.extract_values().values())).reshape(-1, 24)
    z_down = np.array(list(instance.z_down.extract_values().values())).reshape(-1, 24)
    y_down = np.array(list(instance.y_down.extract_values().values())).reshape(-1, 24)
    u_down = np.array(list(instance.u_down.extract_values().values())).reshape(-1, 24)
    p_down = np.array(list(instance.p_down.extract_values().values())).reshape(-1, 24)
    t_c_data = np.array(list(instance.t_c_data.extract_values().values())).reshape(-1)
    t_c = np.array(list(instance.t_c.extract_values().values())).reshape(-1, 96)
    t_c_base = np.array(list(instance.t_c_base.extract_values().values())).reshape(-1)
    t_f = np.array(list(instance.t_f.extract_values().values())).reshape(-1, 96)
    t_f_base = np.array(list(instance.t_f_base.extract_values().values())).reshape(-1)
    lambda_spot = np.array(
        list(instance.lambda_spot.extract_values().values())
    ).reshape(-1, 24)

    s = np.array(list(instance.s.extract_values().values())).reshape(-1, 24)
    slack = s.copy()
    up_regulation_event = np.array(
        list(instance.up_regulation_event.extract_values().values())
    ).reshape(-1, 24)
    g_indicator = np.array(list(instance.g.extract_values().values())).reshape(-1, 24)
    lambda_b = np.array(list(instance.lambda_b.extract_values().values())).reshape(
        -1, 24
    )
    alpha = value(instance.alpha)
    beta = value(instance.beta)
    lambda_rp = np.array(list(instance.lambda_rp.extract_values().values())).reshape(
        -1, 24
    )

    return InstanceInformation(
        p_up_reserve=p_up_reserve,
        p_base=p_base,
        pt=pt,
        s=s,
        slack=slack,
        p_up=p_up,
        u_up=u_up,
        y_up=y_up,
        z_up=z_up,
        z_down=z_down,
        y_down=y_down,
        u_down=u_down,
        p_down=p_down,
        t_c_data=t_c_data,
        t_c=t_c,
        t_c_base=t_c_base,
        t_f=t_f,
        t_f_base=t_f_base,
        up_regulation_event=up_regulation_event,
        g_indicator=g_indicator,
        lambda_b=lambda_b,
        lambda_rp=lambda_rp,
        lambda_spot=lambda_spot,
        alpha=alpha,
        beta=beta,
    )


class Problem:
    def __init__(
        self,
        abstract_model_instance: SolverInstance,
        instance: OptimizationInstance,
    ) -> None:
        data = SolverInstance.instance_to_dict(instance)
        self.model_instance = abstract_model_instance.model.create_instance(data=data)
        self.res_instance: Optional[Any] = None

    def set_objective(self, objective_function: Callable) -> None:
        self.model_instance.objective = Objective(
            rule=objective_function, sense=maximize
        )

    @staticmethod
    def customize_constraints(inst: AbstractModel, one_lambda: bool) -> None:
        if one_lambda:
            inst.lambda_policy_1.deactivate()
            inst.one_lambda_constraint_1.activate()
            inst.one_lambda_constraint_2.activate()
        else:
            inst.lambda_policy_1.activate()
            inst.one_lambda_constraint_1.deactivate()
            inst.one_lambda_constraint_2.deactivate()

    def solve(self, tee: bool = True) -> OptimizationResult:
        self.res_instance, _ = SolverInstance.run_solver(self.model_instance, tee=tee)
        opt_result = self.get_result(self.res_instance, if_print=tee)
        return opt_result

    @staticmethod
    def get_result(
        res_instance: AbstractModel, multiplier: float = 1, if_print: bool = True
    ) -> OptimizationResult:

        p_base = np.array([value(res_instance.p_base[i]) for i in range(24)])
        reserve_payment = (
            value(o_expected_reserve_payment(res_instance)) * multiplier  # type:ignore
        )
        act_payment = (
            value(o_expected_activation_payment(res_instance))  # type:ignore
            * multiplier
        )
        expected_energy_cost = (
            value(o_expected_energy_cost(res_instance)) * multiplier  # type:ignore
        )
        rebound_cost = (
            value(o_expected_rebound_cost(res_instance)) * multiplier  # type:ignore
        )
        expected_power_usage = (
            value(o_expected_energy_consumption(res_instance))  # type:ignore
            * multiplier
        ) / 1000  # mwh
        base_power_usage = (sum(p_base)) * multiplier / 1000  # mwh
        total_cost = -value(o_rule(res_instance)) * multiplier  # type:ignore

        base_cost_today = (
            value(
                sum(  # type:ignore
                    p_base[t]  # type:ignore
                    * (  # type:ignore
                        res_instance.lambda_spot[w, t]  # type:ignore
                        + res_instance.elafgift  # type:ignore
                        + res_instance.tariff[t]  # type:ignore
                    )
                    * (1 + res_instance.moms)  # type:ignore
                    * res_instance.probabilities[w]  # type:ignore
                    for t in res_instance.n_hours  # type:ignore
                    for w in res_instance.nb_scenarios  # type:ignore
                )
            )
            * multiplier
        )
        penalty_cost = value(o_penalty(res_instance)) * multiplier  # type:ignore

        if if_print:
            # print out statistics on earnings/cost
            print(f"Earnings from mFRR reserve: {round(reserve_payment, 1)} DKK")
            print(f"Earnings from mFRR activation: {round(act_payment, 1)} DKK")
            print(f"Earnings from mFRR: {round(reserve_payment + act_payment, 1)} DKK")
            print(f"Base energy usage: {round(base_power_usage, 2)} MWH")
            print(f"Expected energy usage: {round(expected_power_usage, 2)} MWH")
            print(f"Base energy costs today: {round(base_cost_today, 1)} DKK")
            print(f"Expected energy costs: {round(expected_energy_cost, 1)} DKK")
            print(f"Expected total costs: {round(total_cost, 1)} DKK")
            print(f"Expected penalty costs: {round(penalty_cost, 1)} DKK")

            _lambda_b = np.array(
                list(res_instance.lambda_b.extract_values().values())
            ).reshape(-1, 24)
            for i in range(_lambda_b.shape[0]):
                lambda_b = [round(e, 2) if e else 0 for e in _lambda_b[i, :].tolist()]
                print(f"Bid policy: {lambda_b} DKK/kWh")
            p_up_reserve = [
                round(e, 2) for e in res_instance.p_up_reserve.extract_values().values()
            ]
            print(f"p_up reserve policy: {p_up_reserve} kWh")

            print("\n")

        return OptimizationResult(
            reserve_payment=reserve_payment,
            act_payment=act_payment,
            expected_energy_cost=expected_energy_cost,
            rebound_cost=rebound_cost,
            total_cost=total_cost,
            base_cost_today=base_cost_today,
            penalty_cost=penalty_cost,
            battery_capacity=-1,
        )
