import argparse
from abc import ABCMeta
from argparse import ArgumentParser
from typing import List

ETL_REGISTRY = {}  # Map of etl class name and class (must be an ETLComponent)


class BaseETLComponent(ABCMeta):
    def __new__(mcls, name, bases, attrs, **kwargs):  # type: ignore
        new_class = super().__new__(mcls, name, bases, attrs, **kwargs)  # type: ignore

        # Exclude the Component class (if we have not extended)
        parents = [b for b in bases if isinstance(b, BaseETLComponent)]
        if not parents:
            return new_class

        ETL_REGISTRY[name] = new_class

        return new_class


class ETLComponent(metaclass=BaseETLComponent):
    def get_choices(self) -> List[str]:
        choices = []
        for k in self.__dir__():
            if not k.startswith("_:"):
                if "experiment" in k:
                    choices.append(k)
        return choices

    @classmethod
    def add_arguments(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--dry-run", default=False, action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "--overwrite", default=False, action=argparse.BooleanOptionalAction
        )
        parser.add_argument("callable", choices=cls().get_choices())
