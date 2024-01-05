import logging
from typing import Any

from ...experiment_manager.core import ETLComponent

logger = logging.getLogger(__name__)


def some_func(partition: str, **kwargs: Any) -> Any:
    logger.info("Hello world 2")
    return None


class TestExperiment2(ETLComponent):
    def backfill(self, **kwargs: Any) -> None:
        for year in [2020, 2021, 2022]:
            logger.info(f"Year: {year}")
            partition = f"year={year}"
            some_func(partition, **kwargs)
