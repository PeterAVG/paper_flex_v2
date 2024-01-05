import multiprocessing
import os

if True:
    multiprocessing.set_start_method("spawn", True)
import concurrent.futures
from typing import Any

import numpy as np
from pyomo.environ import Objective, maximize

from src.prepare_problem import Problem, get_variables_and_params
from src.problem import SolverInstance


def _solve_one_day(
    payload: Any,
) -> Any:
    is_data, oos_data, objective, one_lambda = payload

    # solve in-sample problem
    generic_solver_instance = SolverInstance()
    is_inst = generic_solver_instance.model.create_instance(data=is_data)

    is_inst.objective = Objective(rule=objective(is_inst), sense=maximize)
    if not one_lambda:
        is_inst.lambda_policy_1.activate()
        is_inst.one_lambda_constraint_1.deactivate()
        is_inst.one_lambda_constraint_2.deactivate()
    else:
        is_inst.lambda_policy_1.deactivate()
        is_inst.one_lambda_constraint_1.activate()
        is_inst.one_lambda_constraint_2.activate()
    SolverInstance.run_solver(is_inst, True)
    is_result = (
        get_variables_and_params(is_inst),
        Problem.get_result(is_inst, if_print=False),
        is_inst.objective.expr(),
    )

    # solve out-of-sample problem using policy from is-sample problem
    generic_solver_instance = SolverInstance()
    oos_inst = generic_solver_instance.model.create_instance(data=oos_data)
    oos_inst.objective = Objective(rule=objective(oos_inst), sense=maximize)
    lambda_spot = np.array(list(oos_inst.lambda_spot.extract_values().values()))
    p_up_reserve = is_result[0].p_up_reserve
    oos_inst.max_lambda_bid = 15

    if not one_lambda:
        oos_inst.lambda_policy_1.activate()
        oos_inst.one_lambda_constraint_1.deactivate()
        oos_inst.one_lambda_constraint_2.deactivate()
        alpha = is_result[0].alpha
        beta = is_result[0].beta
        lambda_bid = np.diff(lambda_spot, append=5) * alpha + beta
        if (lambda_bid < 0).any():
            beta += abs(np.min(lambda_bid)) + 1e-4
        oos_inst.alpha.fix(alpha)
        oos_inst.beta.fix(beta)
    else:
        lambda_b = is_result[0].lambda_b
        oos_inst.lambda_policy_1.deactivate()
        oos_inst.one_lambda_constraint_1.activate()
        oos_inst.one_lambda_constraint_2.activate()

    for i in range(24):
        # fix decision variables related to offering strategy
        v = p_up_reserve[i]
        oos_inst.p_up_reserve[i].fix(v)
        if one_lambda:
            oos_inst.lambda_b[0, i].fix(lambda_b[0, i])  # type:ignore
    try:
        SolverInstance.run_solver(oos_inst, True)
    except Exception as e:
        print(f"Solver failed: {e}")
        raise Exception(
            f"Solver failed: {e}, alpha={alpha}, beta={beta}"  # type:ignore
        )
    oos_result = (
        get_variables_and_params(oos_inst),
        Problem.get_result(oos_inst, if_print=True),
        oos_inst.objective.expr(),
    )

    return is_result, oos_result


def solve_receding_horizon_problem(all_data: Any) -> Any:
    if False:
        max_workers = os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            gen = executor.map(_solve_one_day, all_data)
    else:
        gen = map(_solve_one_day, all_data)

    is_opt_result_list = []
    oos_opt_result_list = []
    for omega, (is_res, oos_res) in enumerate(gen):
        (is_instance_info, is_opt_result, is_objective_val) = is_res
        (oos_instance_info, oos_opt_result, oos_objective_val) = oos_res
        is_opt_result_list.append(is_opt_result)
        oos_opt_result_list.append(oos_opt_result)

    # is_avg_opt_result = average_opt_results(is_opt_result_list)
    # oos_avg_opt_result = average_opt_results(oos_opt_result_list)

    # return is_avg_opt_result, oos_avg_opt_result
    return is_opt_result_list, oos_opt_result_list
