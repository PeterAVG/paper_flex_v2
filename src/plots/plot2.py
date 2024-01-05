#%% # noqa
import datetime
from copy import deepcopy
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.base import Case
from src.common.utils import _set_font_size
from src.evaluation import average_opt_results
from src.experiment_manager.cache import load_cache

# matplotlib.pyplot.ion()

sns.set_theme()
sns.set(font_scale=1.5)


def plot_spot_case_result() -> None:
    cache = load_cache()

    params = Case.default_params()
    params["case"] = Case.SPOT.name
    params["run_oos"] = False
    params["year"] = 2021
    params["one_lambda"] = False  # does not make sense for spot. Delete

    information, _ = cache[params.__repr__()]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(11, 13))
    ax = ax.ravel()
    x = np.array(list(range(24)))
    y = information.p_base - information.p_up_reserve

    # Temperature dynamics
    w = -1
    x2 = np.array(list(range(96))) / 4
    ax[1].plot(x2, information.t_c[w, :], label=r"$T^{c}_{t}$")
    ax[1].plot(x2, information.t_f[w, :], label=r"$T^{f}_{t}$")
    # ax[1].plot(x2, information.t_c_base, label=r"$T^{c}_{t}$", alpha=0.5)
    # ax[1].plot(x2, information.t_f_base, label=r"$T^{f}_{t}$", alpha=0.5)
    ax[1].plot(
        x2, information.t_c_data, label="Measurements", color="black", linestyle=":"
    )
    ax[1].set_ylabel(r"Temperature [$^\circ$C]")
    ax[1].legend(loc="upper right")
    # ax[1].set_xlabel("Time [h]")

    # power dynamics
    _pt = information.pt[w, :]

    ax[0].step(x, information.p_base, label=r"$P^{B}_{h}$", color="black", where="post")
    ax[0].step(x, _pt, label=r"$p_{h}$", color="black", linestyle="--", where="post")
    # ax[0].set_xlabel("Time [h]")
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend(loc="upper left")

    ax[2].step(
        x,
        information.lambda_spot[w, :],
        label=r"$\lambda_{h}^{s}$",
        color="red",
        where="post",
    )
    ax[2].legend(loc="best")
    ax[2].set_ylabel("Price [DKK/kWh]")
    ax[2].set_xlabel("Time [h]")

    # plt.rcParams.update({"font.size": 14})
    _set_font_size(ax, legend=20)
    plt.tight_layout()

    plt.savefig("tex/figures/spot_single_case", dpi=300)


def plot_mFRR_case_result() -> None:
    cache = load_cache()

    params = Case.default_params()
    params["case"] = Case.mFRR_AND_ENERGY.name
    params["run_oos"] = False
    params["year"] = 2021
    params["one_lambda"] = False
    params["admm"] = False
    params["nb_scenarios_spot"] = 10

    information, _ = cache[params.__repr__()]

    fig, ax = plt.subplots(4, 1, figsize=(11, 16))
    ax = ax.ravel()
    x = np.array(list(range(24)))
    y = information.p_base - information.p_up_reserve
    ax[0].step(x, information.p_base, color="black", where="post")
    ax[0].step(x, y, color="black", linestyle="--", where="post")
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend(
        [r"$P^{Base}_{h}$", r"$P^{Base}_{h} - p^{r,\uparrow}_{h}$"],
        loc="best",
    )

    # Temperature dynamics
    w = -1
    # _w = information.lambda_rp.shape[0]
    x2 = np.array(list(range(96))) / 4
    # ax[1].set_title("Scenario {}".format(w))
    ax[1].plot(x2, information.t_c[w, :], label=r"$T^{c}_{t}$")
    ax[1].plot(x2, information.t_f[w, :], label=r"$T^{f}_{t}$")
    # ax[1].plot(x2, information.t_c_base, label=r"$T^{c}_{t}$", alpha=0.5)
    # ax[1].plot(x2, information.t_f_base, label=r"$T^{f}_{t}$", alpha=0.5)
    ax[1].plot(
        x2, information.t_c_data, label="Measurements", color="black", linestyle=":"
    )
    # ax[1].plot(x2, t_f[w, :], label="TcFood")
    ax[1].set_ylabel(r"Temperature [$^\circ$C]")
    ax[1].legend(loc="best")
    # ax[1].set_xlabel("Time [h]")

    # power dynamics
    _pt = information.pt[w, :]

    ax[2].step(
        x,
        information.p_base,
        label=r"$P^{Base}_{h}$",
        color="black",
        where="post",
    )
    ax[2].step(
        x,
        _pt,
        label=r"$P^{Base}_{h} - p^{b,\uparrow}_{h} + p^{b,\downarrow}_{h}$",
        color="black",
        linestyle="--",
        where="post",
    )
    # ax[2].set_xlabel("Time [h]")
    ax[2].set_ylabel("Power [kW]")
    # ax[2].set_title("Scenario {}".format(w))
    ax[2].legend(loc="best")

    ax[3].step(
        x,
        information.lambda_spot[w, :],
        label=r"$\lambda_{h}^{s}$",
        color="red",
        where="post",
    )

    ax[3].step(
        x,
        information.lambda_rp[w, :],
        label=r"$\lambda_{h}^{b}$",
        color="blue",
        where="post",
    )
    ax[3].step(
        x,
        information.lambda_spot[w, :] + information.lambda_b[w, :],
        label=r"$\lambda_{h}^{bid} + \lambda_{h}^{s}$",
        color="orange",
        where="post",
    )
    ax[3].legend(loc="best")
    ax[3].set_ylabel("Price [DKK/kWh]")
    ax[3].set_xlabel("Time [h]")
    # plt.rcParams.update({"font.size": 20})
    _set_font_size(ax, legend=20)
    plt.tight_layout()
    plt.savefig("tex/figures/mFRR_single_case", dpi=300)


def admm_vs_normal_solution() -> None:
    cache = load_cache()

    def verify(params: Dict[str, Any]) -> bool:
        return (
            params.get("run_oos", False) is False
            and Case.mFRR_AND_ENERGY.name in params["case"]
            and params["analysis"] == "analysis1"
            and params["year"] == 2021
            and params["one_lambda"] is False
            and params["nb_scenarios_spot"] in [5, 10]
            and params.get("gamma", -1) in [-1, 10]
        )

    all_params = [deepcopy(eval(p)) for p, _ in cache.items() if verify(eval(p))]

    assert len(all_params) == 4, "We expect 4 results"

    try:
        info_res = [
            {
                "admm": p["admm"],
                "nb": p["nb_scenarios_spot"],
                "info": cache[p.__repr__()][-2],
                "result": cache[p.__repr__()][-1],
            }
            for p in all_params
        ]
    except FileNotFoundError:
        print("File not found.")
        raise Exception("File not found.")

    nb = 5
    admm = [e for e in info_res if e["admm"] and e["nb"] == nb][0]
    normal = [e for e in info_res if (not e["admm"] and e["nb"] == nb)][0]

    # print objective value for each of them
    print(f"Base cost: {admm['result'].base_cost_today}")
    print(f"ADMM results: {admm['result']}")
    print(f"Normal results: {normal['result']}")

    # print p_up_reserve for admm
    p_up_reserve_admm = admm["info"].p_up_reserve.round(2)
    p_up_reserve_normal = normal["info"].p_up_reserve.round(2)
    print(f"ADMM ({nb}): {p_up_reserve_admm}")
    print(f"Normal ({nb}): {p_up_reserve_normal}")

    # print penalty for admm
    # penalty_cost_admm = admm["info"].slack.round(2)
    # penalty_cost_normal = normal["info"].slack.round(2)
    # print(f"ADMM ({nb}): {penalty_cost_admm}")
    # print(f"Normal ({nb}): {penalty_cost_normal}")

    # print lambda alpha
    lambda_alpha_admm = admm["info"].alpha
    lambda_alpha_normal = normal["info"].alpha
    print(f"ADMM ({nb}): {lambda_alpha_admm}")
    print(f"Normal ({nb}): {lambda_alpha_normal}")

    # print lambda beta
    lambda_beta_admm = admm["info"].beta
    lambda_beta_normal = normal["info"].beta
    print(f"ADMM ({nb}): {lambda_beta_admm}")
    print(f"Normal ({nb}): {lambda_beta_normal}")

    def verify2(params: Dict[str, Any]) -> bool:
        return (
            params.get("save_admm_iterations") is not None
            and params["admm"]
            and params["nb_scenarios_spot"] == 5
            and not params.get("run_oos", False)
        )

    gamma_experiments = [(eval(p), v) for p, v in cache.items() if verify2(eval(p))]
    gamma_experiments = sorted(gamma_experiments, key=lambda x: x[0]["gamma"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    # ax = ax.ravel()
    markers = ["o", "s", "v", "d", "p", "h"]
    linestyles = ["-", "--", "-.", ":", "dashdot", "dotted"]
    for i, (param, val) in enumerate(gamma_experiments):
        gamma = param["gamma"]
        total_cost = [e.total_cost for e in val[0]]
        x = np.array(list(range(len(total_cost))))
        ax.plot(
            x,
            total_cost,
            label=rf"$\gamma={gamma}$",
            linestyle=linestyles[i],
            linewidth=1,
            marker=markers[i],
        )
    ax.step(
        x,
        [normal["result"].total_cost for _ in range(len(x))],
        label="Optimal",
        where="post",
        color="black",
        linewidth=1.8,
    )

    ax.set_xlabel("# ADMM iterations")
    ax.set_ylabel("Total cost [DKK]")
    ax.legend(loc="best")
    _set_font_size(ax, legend=20)
    plt.tight_layout()
    plt.savefig("tex/figures/admm_vs_normal_solution.png", dpi=300)


def admm_50_scenarios_convergence() -> None:
    cache = load_cache()

    def verify(params: Dict[str, Any]) -> bool:
        return (
            params["analysis"] == "analysis1"
            and params.get("run_oos") is False
            and Case.mFRR_AND_ENERGY.name in params["case"]
            and (params["year"] == 2021)
            and params["one_lambda"] is False
            and not params.get("save_admm_iterations")
            # and not params.get("gamma")
            and params["admm"]
            and params["nb_scenarios_spot"] == 50
        )

    admm_experiments = [(eval(p), v) for p, v in cache.items() if verify(eval(p))]
    print(len(admm_experiments))

    # TODO: run admm with 50 scenarios where iterations are saved
    exp = admm_experiments[0]


# admm_50_scenarios_convergence()


def admm_scenarios() -> None:
    cache = load_cache()

    def verify(params: Dict[str, Any], oos: bool) -> bool:
        return (
            params["analysis"] == "analysis1"
            and params.get("run_oos") is oos
            and Case.mFRR_AND_ENERGY.name in params["case"]
            and (params["year"] == 2021 if not oos else params["year"] == 2022)
            and params["one_lambda"] is False
            and params["admm"]
            and params["gamma"] == 0.5
        )

    is_admm_experiments = [
        (eval(p), v) for p, v in cache.items() if verify(eval(p), False)
    ]
    is_admm_experiments = sorted(
        is_admm_experiments, key=lambda x: x[0]["nb_scenarios_spot"]
    )
    oos_admm_experiments = [
        (eval(p), v) for p, v in cache.items() if verify(eval(p), True)
    ]
    oos_admm_experiments = sorted(
        oos_admm_experiments, key=lambda x: x[0]["nb_scenarios_spot"]
    )

    assert len(is_admm_experiments) == len(
        oos_admm_experiments
    ), "We expect the same number of experiments"

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    is_nb = [p["nb_scenarios_spot"] for p, _ in is_admm_experiments]
    is_total_cost = [v[-1].total_cost for _, v in is_admm_experiments]
    oos_nb = [p["nb_scenarios_spot"] for p, _ in oos_admm_experiments]
    oos_total_cost = [
        v[1].total_cost
        if not isinstance(v[1], list)
        else average_opt_results(v[1]).total_cost
        for _, v in oos_admm_experiments
    ]

    is_base_cost = [v[-1].base_cost_today for _, v in is_admm_experiments]
    oos_base_cost = [
        v[1].base_cost_today
        if not isinstance(v[1], list)
        else average_opt_results(v[1]).base_cost_today
        for _, v in oos_admm_experiments
    ]

    ax.plot(
        is_nb,
        is_total_cost,
        label="In-sample",
        linestyle="-",
        linewidth=1,
        marker="o",
        color="red",
    )
    ax.plot(
        oos_nb,
        oos_total_cost,
        label="Out-of-sample",
        linestyle="--",
        linewidth=1,
        marker="s",
        color="red",
    )
    ax.plot(
        is_nb,
        is_base_cost,
        label="Base cost (IS)",
        linestyle="-",
        linewidth=1.8,
        color="black",
    )
    ax.plot(
        oos_nb,
        oos_base_cost,
        label="Base cost (OOS)",
        linestyle="--",
        linewidth=1.8,
        color="black",
    )

    ax.set_xlabel("# training scenarios")
    ax.set_ylabel("Total cost [DKK]")
    ax.legend(loc="best")
    _set_font_size(ax, legend=20)
    plt.tight_layout()
    plt.savefig("tex/figures/admm_nb_scenarios_effect.png", dpi=300)


def receding_horizon_scenarios() -> None:
    def verify(params: Dict[str, Any]) -> bool:
        return (
            params["analysis"] == "analysis3"
            and params.get("lookback")
            and params["lookback"] == 5
        )

    def verify_admm(params: Dict[str, Any], nb: int) -> bool:
        return (
            params["analysis"] == "analysis1"
            and params.get("run_oos") is True
            and Case.mFRR_AND_ENERGY.name in params["case"]
            and params["year"] == 2022
            and params["one_lambda"] is False
            and params["admm"]
            and params["nb_scenarios_spot"] == nb
            and params["gamma"] == 0.5
        )

    def verify_oracle(params: Dict[str, Any]) -> bool:
        return params["analysis"] == "analysis4"

    cache = load_cache()
    _lookbacks = [(eval(p), v) for p, v in cache.items() if verify(eval(p))]
    assert len(_lookbacks) == 1, "We expect only one experiment (lookback)"
    experiments = _lookbacks[0]

    is_cost = [v.total_cost for v in experiments[1][0]]
    oos_cost = [v.total_cost for v in experiments[1][1]]
    oos_base_cost = [v.base_cost_today for v in experiments[1][1]]

    nb = 50
    _admm_nb = [(eval(p), v) for p, v in cache.items() if verify_admm(eval(p), nb)]
    assert len(_admm_nb) == 1, "We expect only one experiment"
    oos_admm_nb = [e.total_cost for e in _admm_nb[0][1][1]]

    oracle = [(eval(p), v) for p, v in cache.items() if verify_oracle(eval(p))]
    oracle_total_cost = [e.total_cost for e in oracle[0][1][1]]
    assert len(oracle_total_cost) == len(oos_admm_nb)

    def verify_spot(params: Dict[str, Any]) -> bool:
        return (
            params["analysis"] == "analysis1"
            and params.get("run_oos") is True
            and Case.SPOT.name in params["case"]
            and params["year"] == 2022
            and not params.get("gamma", False)
        )

    _spot = [(eval(p), v) for p, v in cache.items() if verify_spot(eval(p))]
    oos_spot = [e.total_cost for e in _spot[0][1][1]]

    days = list(range(1, len(is_cost) + 1))
    # convert list of integers 'days' to datetimes:
    days = [datetime.datetime(2022, 1, 1) + datetime.timedelta(days=d) for d in days]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(
        days,
        np.cumsum(oos_cost),
        label="mFRR: 5 days lookback",
        linestyle="-",
        linewidth=1,
        marker="o",
        markersize=1,
        color="red",
    )
    ax.plot(
        days,
        np.cumsum(oos_admm_nb),
        label="mFRR: 2021",
        linestyle="-",
        linewidth=1,
        marker="s",
        markersize=1,
        color="blue",
    )
    ax.plot(
        days,
        np.cumsum(oos_spot),
        label="Load shifting",
        linestyle="-",
        linewidth=1,
        marker="v",
        markersize=1,
        color="orange",
    )
    ax.plot(
        days,
        np.cumsum(oos_base_cost),
        label="Base cost",
        linestyle="-",
        linewidth=1.8,
        color="black",
    )
    ax.plot(
        days,
        np.cumsum(oracle_total_cost),
        label="mFRR oracle",
        linestyle="--",
        linewidth=1.2,
        color="black",
    )

    ax.set_ylabel("Cumulative cost of operation [DKK]")
    ax.legend(loc="best")
    ax.xaxis.set_tick_params(rotation=45)
    _set_font_size(ax, legend=18)
    plt.tight_layout()
    plt.savefig("tex/figures/cumulative_cost_comparison.png", dpi=300)
    plt.show()

    pass


def main() -> None:
    admm_vs_normal_solution()
    admm_scenarios()
    receding_horizon_scenarios()
    plot_spot_case_result()
    plot_mFRR_case_result()


if __name__ == "__main__":
    main()
