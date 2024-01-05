import multiprocessing
from copy import deepcopy

if True:
    multiprocessing.set_start_method("spawn", True)
import concurrent.futures
from dataclasses import dataclass
from time import time
from typing import Any, Callable, List, Tuple, cast

import numpy as np
from pyomo.environ import Objective, maximize

from src.evaluation import average_opt_results
from src.objective import admm_objective_function
from src.prepare_problem import (
    InstanceInformation,
    OptimizationResult,
    Problem,
    get_variables_and_params,
)
from src.problem import OptimizationInstance, SolverInstance


@dataclass
class Iteration:
    """Dataclass for storing the results of an iteration of ADMM."""

    iteration: int
    cost: List[float]
    total_cost: float
    conv1: float
    conv2: float
    conv3: float
    bar_p_up_reserve: np.ndarray
    bar_alpha: float
    bar_beta: float
    prev_p_up_reserve: np.ndarray
    prev_alpha: np.ndarray
    prev_beta: np.ndarray

    @property
    def distance(self) -> float:
        return self.conv1 + self.conv2 + self.conv3

    @property
    def mean_cost(self) -> float:
        return cast(float, np.mean(self.cost))

    @property
    def std_cost(self) -> float:
        return cast(float, np.std(self.cost))

    def converged(self, tol: float) -> bool:
        return self.distance < tol


def _solve_func(payload: Any) -> Tuple[InstanceInformation, OptimizationResult, float]:
    data, objective, gamma, rho, omega, bar_p_up_reserve, bar_alpha, bar_beta = payload
    generic_solver_instance = SolverInstance()
    inst = generic_solver_instance.model.create_instance(data=data)
    total_objective = objective(inst) + admm_objective_function(
        inst,
        gamma,
        rho,
        omega,
        bar_p_up_reserve,
        bar_alpha,
        bar_beta,
    )
    # total_objective = objective(inst)
    inst.objective = Objective(rule=total_objective, sense=maximize)
    inst.lambda_policy_1.activate()
    inst.one_lambda_constraint_1.deactivate()
    inst.one_lambda_constraint_2.deactivate()
    # inst.alpha.fix(0)
    # inst.beta.fix(0.001)
    # optimal_reserve = [
    #     0.0,
    #     0.77,
    #     0.0,
    #     0.78,
    #     0.8,
    #     0.79,
    #     1.09,
    #     0.0,
    #     0.0,
    #     1.28,
    #     1.28,
    #     1.29,
    #     1.29,
    #     1.26,
    #     1.27,
    #     1.27,
    #     1.27,
    #     1.26,
    #     1.23,
    #     1.17,
    #     1.16,
    #     1.05,
    #     0.86,
    #     0.8,
    # ]
    # for i in range(24):
    #     inst.p_up_reserve[i].fix(max(optimal_reserve[i] - 0.01, 0))
    SolverInstance.run_solver(inst, False)
    return (
        get_variables_and_params(inst),
        Problem.get_result(inst, if_print=False),
        # Problem.get_result(inst, if_print=True),
        inst.objective.expr(),
    )


class ADMM:
    def __init__(
        self,
        generic_solver_instance: SolverInstance,
        instance: OptimizationInstance,
        objective: Callable,
        gamma: float = 1,
        max_iter: int = 20,
        min_iter: int = 3,
        tol: float = 0.1,
    ):
        self.generic_solver_instance = generic_solver_instance
        self.instance = instance
        self.nb_scenarios = instance.nb_scenarios
        self.objective = objective

        self.gamma = gamma
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol

        print(
            f"ADMM: scenarios={self.nb_scenarios}, gamma={gamma}, max_iter={max_iter}, tol={tol}"
        )

    def calculate_error(self, a: np.ndarray, b: np.ndarray) -> float:
        # assert a.shape == b.shape
        # return np.sum(np.sqrt((np.mean((a - b) ** 2, axis=0))))
        return np.sum(np.mean(abs(a - b), axis=0))

    def solve_sweep(
        self,
        bar_p_up_reserve: np.ndarray,
        bar_alpha: float,
        bar_beta: float,
        rho: List[np.ndarray],
        prev_p_up_reserve: np.ndarray,
        prev_alpha: np.ndarray,
        prev_beta: np.ndarray,
        iteration_cost: List[float],
        iteration_opt_result: List[OptimizationResult],
    ) -> None:

        start = time()
        # no gamme penalty in the first iteration
        gamma = self.gamma if len(iteration_cost) > 0 else 0
        all_data = []
        for omega in range(self.nb_scenarios):
            all_data.append(
                (
                    SolverInstance.instance_to_dict(self.instance(omega, omega + 1)),
                    self.objective,
                    gamma,
                    rho,
                    omega,
                    bar_p_up_reserve,
                    bar_alpha,
                    bar_beta,
                )
            )

        if self.nb_scenarios >= 20:
            # Solve in parallel if there are many scenarios
            with concurrent.futures.ProcessPoolExecutor() as executor:
                gen = executor.map(_solve_func, all_data)
        else:
            # Solve sequentially if there are few scenarios
            gen = map(_solve_func, all_data)

        # _data = [
        #     SolverInstance.instance_to_dict(self.instance),
        #     self.objective,
        #     0,
        #     rho,
        #     0,
        #     bar_p_up_reserve,
        #     bar_alpha,
        #     bar_beta,
        # ]
        # normal_solve = _solve_func(_data)

        for omega, (instance_info, opt_result, objective_val) in enumerate(gen):
            # Get variable results
            _p_up_reserve = instance_info.p_up_reserve
            _alpha = instance_info.alpha
            _beta = instance_info.beta

            # assert opt_result.total_cost <= opt_result.base_cost_today

            iteration_cost.append(objective_val)
            iteration_opt_result.append(opt_result)

            # Store optimization results for this scenario
            prev_p_up_reserve[omega, :] = _p_up_reserve
            prev_alpha[omega] = _alpha
            prev_beta[omega] = _beta

        print(f"Solving sweep took {time() - start} seconds")

    def run_admm(self) -> List[Iteration]:
        # Initialize and run ADMM
        # NOTE: we assume equal probability for each scenario
        # TODO: make admm work with one lambda_bid as well

        # Specifiy consensus variables for ADMM
        prev_p_up_reserve = np.tile(
            self.instance.p_base * (1 - self.instance.mask), (self.nb_scenarios, 1)
        )
        prev_alpha = np.zeros(self.nb_scenarios)
        prev_beta = np.zeros(self.nb_scenarios)
        shape1 = prev_p_up_reserve.shape
        shape2 = prev_beta.shape

        bar_p_up_reserve = np.mean(prev_p_up_reserve, axis=0)
        assert bar_p_up_reserve.shape[0] == self.instance.n_hours
        bar_alpha = np.mean(prev_alpha)
        bar_beta = np.mean(prev_beta)

        # Initialize rho
        rho = [
            np.zeros(shape=shape1),
            np.zeros(shape=shape2),
            np.zeros(shape=shape2),
        ]

        j = 0
        runs = []
        print("Running ADMM...")
        while True:
            # p_up_reserve = deepcopy(prev_p_up_reserve)
            # alpha = deepcopy(prev_alpha)
            # beta = deepcopy(prev_beta)

            iteration_cost = []  # type:ignore
            iteration_opt_result = []  # type:ignore

            self.solve_sweep(
                bar_p_up_reserve,
                cast(float, bar_alpha),
                cast(float, bar_beta),
                rho,
                prev_p_up_reserve,
                prev_alpha,
                prev_beta,
                iteration_cost,
                iteration_opt_result,
            )

            # Recalculate average values, i.e. consensus variable values
            bar_p_up_reserve = np.mean(prev_p_up_reserve, axis=0)
            bar_alpha = np.mean(prev_alpha)
            bar_beta = np.mean(prev_beta)

            print("ADMM In-sample results:")
            print(
                f"bar_p_up_reserve={bar_p_up_reserve.round(2)}, bar_alpha={bar_alpha}, bar_beta={bar_beta}"
            )
            base_cost = np.mean([r.base_cost_today for r in iteration_opt_result])
            total_cost = np.mean([r.total_cost for r in iteration_opt_result])
            reserve_payment = np.mean([r.reserve_payment for r in iteration_opt_result])
            activation_payment = np.mean([r.act_payment for r in iteration_opt_result])
            penalty = np.mean([r.penalty_cost for r in iteration_opt_result])
            print(
                f"reserve_payment={reserve_payment}, activation_payment={activation_payment}, penalty={penalty}"
            )
            print(f"base_cost={base_cost}\ntotal_cost={total_cost}")

            # Update Lagrangian multipliers
            rho[0] = rho[0] + self.gamma * (prev_p_up_reserve - bar_p_up_reserve)
            rho[1] = rho[1] + self.gamma * (prev_alpha - bar_alpha)
            rho[2] = rho[2] + self.gamma * (prev_beta - bar_beta)

            # TODO: distance and convergence criteria is wrong somehow
            # conv1 = self.calculate_error(prev_p_up_reserve, p_up_reserve)
            # conv2 = self.calculate_error(prev_alpha, alpha)
            # conv3 = self.calculate_error(prev_beta, beta)
            conv1 = self.calculate_error(prev_p_up_reserve, bar_p_up_reserve)
            conv2 = self.calculate_error(prev_alpha, bar_alpha)  # type:ignore
            conv3 = self.calculate_error(prev_beta, bar_beta)  # type:ignore

            # save iteration results
            runs.append(
                Iteration(
                    j,
                    iteration_cost,
                    0,
                    conv1,
                    conv2,
                    conv3,
                    bar_p_up_reserve,
                    bar_alpha,  # type:ignore
                    bar_beta,  # type:ignore
                    deepcopy(prev_p_up_reserve),
                    deepcopy(prev_alpha),
                    deepcopy(prev_beta),
                )
            )
            print("True in-sample results:")
            info, opt_res = self.prepare_in_sample_results(runs[-1], tee=False)
            print(
                f"reserve_payment={opt_res.reserve_payment}, activation_payment={opt_res.act_payment}, penalty={opt_res.penalty_cost}"
            )
            print(f"base_cost={opt_res.base_cost_today}")
            print(f"total_cost={opt_res.total_cost}")

            print(
                f"\n\nIteration {j}: distance={round(runs[-1].distance,3)}, total_cost={round(opt_res.total_cost, 2)}\n\n"
            )
            runs[-1].total_cost = opt_res.total_cost

            if (
                runs[-1].converged(self.tol) or j > self.max_iter
            ) and j > self.min_iter:
                print(f"\n\nADMM has converged after {j} iterations\n\n")
                break

            j += 1

        return runs

    def make_p_up_reserve_feasible(self, last_iter: Iteration) -> np.ndarray:
        # Make p_up_reserve feasible by ensuring it is always less than p_base
        p_up_reserve = last_iter.bar_p_up_reserve
        for i in range(self.instance.n_hours):
            # reservation can't be bigger than baseline power
            p_up_reserve[i] = min(p_up_reserve[i], self.instance.p_base[i])
            p_up_reserve[i] = p_up_reserve[i] * (1 - self.instance.mask[i])
        return p_up_reserve

    def make_lambda_bid_feasible(
        self, instance: OptimizationInstance, last_iter: Iteration
    ) -> float:
        # Make lambda_bid feasible by ensuring it is always non-negative
        # by increasing beta until lambda_bid is non-negative
        alpha = last_iter.bar_alpha
        beta = last_iter.bar_beta

        # our bid policy
        lambda_bid = np.diff(instance.lambda_spot, append=5) * alpha + beta

        if (lambda_bid < 0).any():
            beta += abs(np.min(lambda_bid)) + 1e-4
            # lambda_bid = np.maximum(lambda_bid, 0)
        return beta

    def prepare_in_sample_results(self, last_iter: Iteration, tee: bool = True) -> Any:
        # Get results from last iteration
        # Only used for ADMM where we need to evaluate the in-sample problem
        # using average parameter values.
        # TODO: perhaps parallelize this and move to "evaluation.py"

        # NOTE: we assume equal probability for each scenario
        p_up_reserve = self.make_p_up_reserve_feasible(last_iter)

        # TODO: chunk size > 1 does NOT give same base cost. Something is wrong. Use 1 for now.
        chunk_size = 1
        opt_results = []
        if tee:
            print(f"Preparing in-sample results with chunk size {chunk_size} ...")
        start = time()
        # for omega in range(self.nb_scenarios):
        for i, chunk_instance in enumerate(self.instance.chunk_generator(chunk_size)):
            if tee:
                print(f"   Evaluating chunk {i+1}...")
            chunk_instance.max_lambda_bid = 15
            problem = Problem(self.generic_solver_instance, chunk_instance)
            Problem.customize_constraints(
                problem.model_instance, self.instance.one_lambda
            )
            problem.set_objective(self.objective(problem.model_instance))
            # fix variables
            for i in range(self.instance.n_hours):
                problem.model_instance.p_up_reserve[i].fix(p_up_reserve[i])
            problem.model_instance.alpha.fix(last_iter.bar_alpha)
            beta = self.make_lambda_bid_feasible(chunk_instance, last_iter)
            problem.model_instance.beta.fix(beta)

            opt_result = problem.solve(tee=False)
            opt_results.append(opt_result)
            # inst_info = get_variables_and_params(problem.model_instance)

        avg_opt_result = average_opt_results(opt_results)
        # assert avg_opt_result.total_cost <= avg_opt_result.base_cost_today
        if tee:
            print(f"In-sample results prepared in {time() - start} seconds")

        return (get_variables_and_params(problem.res_instance), avg_opt_result)  # type: ignore
