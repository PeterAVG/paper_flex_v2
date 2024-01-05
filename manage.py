import logging
from argparse import ArgumentParser
from typing import Type

from src.experiment_manager.core import ETL_REGISTRY, ETLComponent

logging.basicConfig(level=logging.INFO)
logging.getLogger("pyomo").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handle_invocation() -> None:
    parser = ArgumentParser("Experiment run:")
    subparser = parser.add_subparsers(required=False)

    for key, klass in ETL_REGISTRY.items():
        etl_parser = subparser.add_parser(key)
        klass.add_arguments(etl_parser)  # type: ignore
        etl_parser.set_defaults(etl_class=klass)

    try:
        args = parser.parse_args()
    except TypeError:
        logger.info("Some arguments are required")
    else:
        Etl: Type[ETLComponent] = args.etl_class
        etl = Etl()
        logger.info(f"Running {Etl.__name__}")
        logger.info(f"Args: {args.__dict__}")
        callable_str = args.callable
        func = getattr(etl, callable_str)
        assert callable(func)
        _args = args.__dict__
        _args.pop("etl_class")
        func(**_args)


if __name__ == "__main__":
    import src.experiments  # noqa

    handle_invocation()

    pass
