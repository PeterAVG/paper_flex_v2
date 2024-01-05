from functools import wraps
from time import time
from typing import Any, Callable, Generator, List


def _set_font_size(ax: Any, misc: int = 26, legend: int = 14) -> None:
    try:
        _ = len(ax)
    except TypeError:
        ax = [ax]
    for _ax in ax:
        for item in (
            [_ax.title, _ax.xaxis.label, _ax.yaxis.label]
            + _ax.get_xticklabels()
            + _ax.get_yticklabels()
        ):
            item.set_fontsize(misc)
    for _ax in ax:
        for item in _ax.get_legend().get_texts():
            item.set_fontsize(legend)


def timing(print_output: bool = False) -> Callable:
    def timing_decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrap(*args: Any, **kw: Any) -> Any:
            ts = time()
            result = f(*args, **kw)
            te = time()
            text = "func:%r took: %2.4f sec" % (f.__name__, te - ts)
            text += f" | Result: {result})" if print_output else ""
            print(text)
            return result

        return wrap

    return timing_decorator


def chunks(lst: List[Any], n: int) -> Generator:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
