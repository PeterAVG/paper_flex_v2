from enum import Enum
from typing import Any, Dict

from src.objective import _o_energy_cost, o_rule, o_rule_no_energy

SCENARIO_PATH = "data/scenarios_v2.csv"


class Case(Enum):
    mFRR_AND_ENERGY = "joint_mfrr_and_energy"
    mFRR = "mfrr_only"
    SPOT = "energy_only"
    NAIVE = "naive"  # No scenarios, rule based approach (mFRR_AND_ENERGY) | NOT USED
    ROBUST = "robust"  # single, worst case scenario (mFRR_AND_ENERGY) | NOT USED

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return {
            "elafgift": 0.0,
            "moms": 0.0,
            "delta_max": 50,
            "analysis": "analysis1",
        }


OBJECTIVE_FUNCTION = {
    Case.mFRR_AND_ENERGY.name: o_rule,
    Case.mFRR.name: o_rule_no_energy,
    Case.SPOT.name: _o_energy_cost,
    Case.NAIVE.name: o_rule,
    Case.ROBUST.name: o_rule,
}
