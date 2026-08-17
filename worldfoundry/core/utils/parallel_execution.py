"""Thread-pool and async helpers for parallel I/O and batch execution."""

import asyncio
import os
from functools import wraps
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import ThreadPool as ProcessThreadPool
from threading import Thread
from typing import Callable, Dict, List

from tqdm import tqdm


def async_call_func(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    return wrapper


def slice_func(chunk_index, chunk_dim, chunk_size):
    """Build an index that selects one chunk along ``chunk_dim``."""
    return [slice(None)] * chunk_dim + [slice(chunk_index, chunk_index + chunk_size)]


def async_call(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()

    return wrapper


def _save_image_impl(save_img, save_path):
    import imageio

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, save_img)


@async_call
def save_image_async(save_img, save_path):
    _save_image_impl(save_img, save_path)


def save_image(save_img, save_path):
    _save_image_impl(save_img, save_path)


def parallel_execution(
    *args,
    action: Callable,
    num_processes=32,
    print_progress=False,
    sequential=False,
    async_return=False,
    desc=None,
    **kwargs,
):
    """Map *action* over zipped *args* and *kwargs* using a THREAD pool.

    Despite the ``num_processes`` name (kept for API compatibility), workers
    are threads (``multiprocessing.pool.ThreadPool``): CPU-bound actions do
    not run in parallel under the GIL; use ``parallel_processes`` for those.
    With ``async_return=True`` the live pool is returned and the caller owns
    it (must ``close()``/``join()`` after retrieving results).
    """
    args = list(args)

    def get_length(args: List, kwargs: Dict):
        for arg in args:
            if isinstance(arg, list):
                return len(arg)
        for value in kwargs.values():
            if isinstance(value, list):
                return len(value)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        action_args = [(arg[i] if isinstance(arg, list) and len(arg) == length else arg) for arg in args]
        action_kwargs = {
            key: (kwargs[key][i] if isinstance(kwargs[key], list) and len(kwargs[key]) == length else kwargs[key])
            for key in kwargs
        }
        return action_args, action_kwargs

    if not sequential:
        pool = ProcessThreadPool(processes=num_processes)
        results = []
        try:
            asyncs = []
            length = get_length(args, kwargs)
            for i in range(length):
                action_args, action_kwargs = get_action_args(length, args, kwargs, i)
                asyncs.append(pool.apply_async(action, action_args, action_kwargs))

            if async_return:
                return pool

            for async_result in tqdm(asyncs, desc=desc, disable=not print_progress):
                results.append(async_result.get())
        except BaseException:
            # Historical behavior leaked the pool when an action raised.
            pool.terminate()
            pool.join()
            raise
        pool.close()
        pool.join()
        return results

    results = []
    length = get_length(args, kwargs)
    for i in tqdm(range(length), desc=desc, disable=not print_progress):
        action_args, action_kwargs = get_action_args(length, args, kwargs, i)
        results.append(action(*action_args, **action_kwargs))
    return results


def parallel_threads(
    function,
    args,
    workers=0,
    star_args=False,
    kw_args=False,
    front_num=1,
    Pool=ThreadPool,
    **tqdm_kw,
):
    while workers <= 0:
        workers += cpu_count()
    if workers == 1:
        front_num = float("inf")

    try:
        n_args_parallel = len(args) - front_num
    except TypeError:
        n_args_parallel = None
    args = iter(args)

    front = []
    while len(front) < front_num:
        try:
            arg = next(args)
        except StopIteration:
            return front
        front.append(function(*arg) if star_args else function(**arg) if kw_args else function(arg))

    out = []
    with Pool(workers) as pool:
        if star_args:
            futures = pool.imap(starcall, [(function, arg) for arg in args])
        elif kw_args:
            futures = pool.imap(starstarcall, [(function, arg) for arg in args])
        else:
            futures = pool.imap(function, args)
        for result in tqdm(futures, total=n_args_parallel, **tqdm_kw):
            out.append(result)
    return front + out


def parallel_processes(*args, **kwargs):
    import multiprocessing as mp

    kwargs["Pool"] = mp.Pool
    return parallel_threads(*args, **kwargs)


def starcall(args):
    function, function_args = args
    return function(*function_args)


def starstarcall(args):
    function, function_args = args
    return function(**function_args)
