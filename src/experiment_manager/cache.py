import os
from functools import wraps
from typing import Any, Callable, Dict

import cloudpickle
from dotenv.main import load_dotenv

load_dotenv()

DEFAULT_EXPERIMENT_FOLDER = "src/experiments"
EXPERIMENT_FOLDER = os.getenv("EXPERIMENT_FOLDER", DEFAULT_EXPERIMENT_FOLDER)
# TODO: create 'dry_runs' and 'cache' folder when not present

CACHE_FILE = f"{EXPERIMENT_FOLDER}/cache/cache.pkl"
DRY_RUN_FILE = f"{EXPERIMENT_FOLDER}/dry_runs/cache.pkl"


def save_to_cache(cache: Any, dry_run: bool = False) -> None:
    FILE = CACHE_FILE if not dry_run else DRY_RUN_FILE
    with open(
        FILE,
        mode="wb",
    ) as file:
        cloudpickle.dump(cache, file)


def load_cache() -> Dict[str, Any]:
    try:
        with open(
            CACHE_FILE,
            mode="rb",
        ) as _file:
            cache = cloudpickle.load(_file)
    except FileNotFoundError:
        print("Cache not found. Initializing empty cache.")
        cache = {}
    return cache


def cache(func: Callable) -> Callable:

    func.cache: Dict[str, Any] = load_cache()  # type:ignore

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        assert len(args) == 1, "We need only one partition"
        partition = args[0]

        dry_run = kwargs["dry_run"]
        overwrite = kwargs["overwrite"]

        if dry_run:
            # run partition but save to dry_run cache
            func.cache[partition] = result = func(partition)  # type:ignore
            save_to_cache(func.cache, dry_run=True)  # type:ignore
            return result
        else:
            if overwrite:
                # run partition and save to cache
                func.cache[partition] = result = func(partition)  # type:ignore
                save_to_cache(func.cache)  # type:ignore
                return result
            else:
                # run partition only if not present in cache, otherwise, return cache
                if func.cache.get(partition) is not None:  # type:ignore
                    return func.cache[partition]  # type:ignore
                else:
                    func.cache[partition] = result = func(  # type:ignore
                        partition
                    )  # type:ignore
                    save_to_cache(func.cache)  # type:ignore
                    return result

    return wrapper
