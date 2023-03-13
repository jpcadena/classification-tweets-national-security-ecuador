"""
Decorator script
"""
import functools
import logging
from time import perf_counter
from typing import Callable, Any

logger: logging.Logger = logging.getLogger(__name__)


def with_logging(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for logging
    :param func: function to be called
    :type func: Callable
    :return: Wrapped function
    :rtype: Any
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info("Calling %s", func.__name__)
        value = func(*args, **kwargs)
        logger.info("Finished %s", func.__name__)
        return value

    return wrapper


def benchmark(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Benchmark function for computational functions
    :param func: Function to be executed
    :type func: Callable
    :return: Wrapped function
    :rtype: Any
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()
        value = func(*args, **kwargs)
        end_time = perf_counter()
        run_time = end_time - start_time
        logger.info("Execution of %s took %s seconds.",
                    func.__name__, run_time)
        return value

    return wrapper
