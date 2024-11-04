import concurrent.futures
from itertools import repeat
from tqdm import tqdm


def parallel_map(f, *args, concurrency=25, use_tqdm=False, **kwargs):
    """
    Parallel map function that supports both args and kwargs.

    :param f: Function to apply
    :param args: Positional arguments to be mapped
    :param concurrency: Number of concurrent threads (default: 25)
    :param use_tqdm: use tqdm to show a progress bar
    :param kwargs: Keyword arguments to be mapped
    :return: List of results
    """
    def ensure_iterable(arg):
        return [arg] if not hasattr(arg, '__iter__') or isinstance(arg, str) else list(arg)

    # Convert args and kwargs to lists
    args = [ensure_iterable(arg) for arg in args]
    kwargs = {k: ensure_iterable(v) for k, v in kwargs.items()}

    # Find the maximum length
    lengths = [len(arg) for arg in args] + [len(v) for v in kwargs.values()]
    if lengths:
        max_length = max(lengths)
    else:
        max_length = concurrency

    # Broadcast length 1 iterables and check lengths
    for i, arg in enumerate(args):
        if len(arg) == 1:
            args[i] = arg * max_length
        elif len(arg) != max_length:
            raise ValueError(f"Argument {i} has length {len(arg)}, expected {max_length}")

    for k, v in kwargs.items():
        if len(v) == 1:
            kwargs[k] = v * max_length
        elif len(v) != max_length:
            raise ValueError(f"Keyword argument '{k}' has length {len(v)}, expected {max_length}")

    # Zip args and kwargs
    zipped_args = zip(*args) if args else repeat((), max_length)
    zipped_kwargs = [dict(zip(kwargs.keys(), kwarg_values)) for kwarg_values in zip(*kwargs.values())] if kwargs else repeat({}, max_length)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(f, *a, **kw) for a, kw in zip(zipped_args, zipped_kwargs)]
        if use_tqdm:
            # Use tqdm to show progress, but don't affect the result order
            list(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"))
        return [future.result() for future in futures]
