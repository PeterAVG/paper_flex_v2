from typing import Any, Callable, List

import numpy as np
from pyomo.environ import AbstractModel


def o_expected_reserve_payment(model: AbstractModel) -> Any:
    return sum(
        [
            model.probabilities[w] * model.p_up_reserve[h] * model.lambda_mfrr[w, h]
            for h in model.n_hours
            for w in model.nb_scenarios
        ]
    )


def o_expected_activation_payment(model: AbstractModel) -> Any:
    return sum(
        [
            model.probabilities[w] * model.p_up[w, h] * model.lambda_rp[w, h]
            for h in model.n_hours
            for w in model.nb_scenarios
        ]
    )


def o_penalty(model: AbstractModel) -> Any:
    return sum(
        [
            model.probabilities[w] * model.s[w, h] * 20000 / 1000
            for h in model.n_hours
            for w in model.nb_scenarios
        ]
    )


def o_expected_extra_cost(model: AbstractModel) -> Any:
    # extra costs from tariffs, moms, etc.
    return sum(
        [
            model.probabilities[w]
            * (
                model.pt[w, h]
                * (model.elafgift + model.tariff[h])
                * (1 + model.moms)  # taxes + 25% moms
                + (
                    model.pt[w, h] * model.lambda_spot[w, h] * model.moms
                )  # 25% moms of spot price
            )
            for h in model.n_hours
            for w in model.nb_scenarios
        ]
    )


def o_expected_rebound_cost(model: AbstractModel) -> Any:
    return sum(
        [
            model.probabilities[w] * model.p_down[w, h] * model.lambda_rp[w, h]
            for h in model.n_hours
            for w in model.nb_scenarios
        ]
    )


def o_expected_energy_cost(model: AbstractModel, extra_cost: bool = True) -> Any:
    # NOTE: we always pay for our baseline, no matter the activation.
    # And then we pay the rebound as well (p_down) as rp-price.
    # The rp-price is equal to the spot price if the systen is not in need of
    # up/down regulation.
    _extra_cost = o_expected_extra_cost(model) if extra_cost else 0
    return (
        sum(
            [
                model.probabilities[w]
                * (
                    model.p_base[h]
                    * model.lambda_spot[w, h]  # cost of buying baseline load
                    # + model.p_down[w, h] * model.lambda_rp[w, h]
                )
                for h in model.n_hours
                for w in model.nb_scenarios
            ]
        )
        + _extra_cost
    )


def o_rule(model: AbstractModel) -> Any:
    return (
        o_expected_reserve_payment(model)
        + o_expected_activation_payment(model)
        - o_penalty(model)
        - o_expected_energy_cost(model)
        - o_expected_rebound_cost(model)
    )


def o_expected_energy_consumption(model: AbstractModel) -> Any:
    return sum(
        [
            model.probabilities[w] * model.pt[w, h]
            for h in model.n_hours
            for w in model.nb_scenarios
        ]
    )


def _o_energy_cost(model: AbstractModel) -> Any:
    return -sum(
        [
            model.pt[w, t]
            * (model.lambda_spot[w, t] + model.elafgift + model.tariff[t])
            * (1 + model.moms)
            * model.probabilities[w]
            for t in model.n_hours
            for w in model.nb_scenarios
        ]
    )


def o_rule_no_energy(model: AbstractModel) -> Any:
    return (
        o_expected_reserve_payment(model)
        + o_expected_activation_payment(model)
        # - o_expected_rebound_cost(model)
        - o_penalty(model)
    )


def admm_objective_function(
    model: AbstractModel,
    gamma: float,
    rho: List[np.ndarray],
    omega: int,
    bar_p_up_reserve: np.ndarray,
    bar_alpha: float,
    bar_beta: float,
) -> Callable:
    def admm_objective_term1(model: AbstractModel) -> Any:
        return -(
            sum(model.p_up_reserve[h] * rho[0][omega, h] for h in model.n_hours)
            + model.alpha * rho[1][omega]
            + model.beta * rho[2][omega]
        )

    def admm_objective_term2(model: AbstractModel) -> Any:
        return (
            -gamma
            / 2
            * (
                sum(
                    (model.p_up_reserve[h] - bar_p_up_reserve[h]) ** 2
                    for h in model.n_hours
                )
                + (model.alpha - bar_alpha) ** 2
                + (model.beta - bar_beta) ** 2
            )
        )

    return admm_objective_term1(model) + admm_objective_term2(model)
