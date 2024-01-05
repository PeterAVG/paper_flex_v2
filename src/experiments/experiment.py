import logging
from typing import Any

from src.optimization import run_optimization

from ..base import Case
from ..experiment_manager.cache import cache  # noqa
from ..experiment_manager.core import ETLComponent

logger = logging.getLogger(__name__)


class LookbackExperiment(ETLComponent):
    def experiment_all_lookback_cases(self, **kwargs: Any) -> None:
        lookbacks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for case in [Case.mFRR_AND_ENERGY]:
            for lookback in lookbacks:
                params = Case.default_params()
                params["analysis"] = "analysis3"
                params["one_lambda"] = False
                params["lookback"] = lookback
                params["case"] = case.name
                params["year"] = 2022

                partition = params.__repr__()
                logger.info(partition, kwargs)

                run_optimization(partition, **kwargs)


class LoadShiftingExperiment(ETLComponent):
    def experiment_spot(self, **kwargs: Any) -> None:
        year = [2021, 2022]
        run_oos = [False, True]
        for _run_oos, _year in zip(run_oos, year):
            params = Case.default_params()
            params["case"] = Case.SPOT.name
            params["run_oos"] = _run_oos
            params["year"] = _year
            params["one_lambda"] = False  # does not matter for SPOT

            partition = params.__repr__()
            logger.info(partition, kwargs)

            run_optimization(partition, **kwargs)


class mFRRExperiment(ETLComponent):
    def experiment_all(self, **kwargs: Any) -> None:
        nb_scenarios = [1, 5, 10, 20, 30, 40, 50, 100, 250]
        year = [2021, 2022]
        run_oos = [False, True]
        admm = [True, False]
        for _nb_scenarios in nb_scenarios:
            for _admm in admm:
                for gamma in [0.5, 10]:
                    for _run_oos, _year in zip(run_oos, year):
                        if not _admm and _nb_scenarios > 20:
                            continue
                        params = Case.default_params()
                        params["case"] = Case.mFRR_AND_ENERGY.name
                        params["run_oos"] = _run_oos
                        params["year"] = _year
                        params["one_lambda"] = False
                        params["admm"] = _admm
                        params["nb_scenarios_spot"] = _nb_scenarios
                        if _admm and not _run_oos:
                            params["save_admm_iterations"] = True
                            params["gamma"] = gamma

                        partition = params.__repr__()
                        logger.info(partition, kwargs)

                        run_optimization(partition, **kwargs)

    def experiment_gamma_admm(self, **kwargs: Any) -> None:
        for gamma in [0.01, 0.1, 0.5, 1.0, 10, 50]:
            # NOTE: order of keys in dict matters a lot here
            params = {
                **Case.default_params(),
                "case": "mFRR_AND_ENERGY",
                "run_oos": False,
                "year": 2021,
                "one_lambda": False,
                "admm": True,
                "nb_scenarios_spot": 5,
                "save_admm_iterations": True,
                "gamma": gamma,
            }

            partition = params.__repr__()
            logger.info(partition, kwargs)

            run_optimization(partition, **kwargs)

    def experiment_oracle(self, **kwargs: Any) -> None:
        params = Case.default_params()
        params["analysis"] = "analysis4"
        params["case"] = Case.mFRR_AND_ENERGY.name
        params["run_oos"] = True
        params["year"] = 2022
        params["one_lambda"] = False
        params["admm"] = False
        params["nb_scenarios_spot"] = 1

        partition = params.__repr__()
        logger.info(partition, kwargs)

        run_optimization(partition, **kwargs)
