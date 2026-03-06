import inspect
import warnings


def get_clean_kwargs(function, warn: bool, kwargs: dict) -> dict:
    # sampler_args = inspect.signature(function).parameters
    # clean_kwargs = {k: kwargs[k] for k in kwargs if k in sampler_args}

    # if warn and set(clean_kwargs) != set(kwargs):
    #     warnings.warn(
    #         f"Ignoring kwargs {set(kwargs).difference(clean_kwargs)} when calling function {function}",
    #         UserWarning,
    #     )

    # return clean_kwargs
    # Edit by Xiao
    params = inspect.signature(function).parameters

    # 如果函数接收 **kwargs（VAR_KEYWORD），则不做过滤，直接返回全部 kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs.copy()

    clean_kwargs = {k: kwargs[k] for k in kwargs if k in params}

    if warn and set(clean_kwargs) != set(kwargs):
        warnings.warn(
            f"Ignoring kwargs {set(kwargs).difference(clean_kwargs)} when calling function {function}",
            UserWarning,
        )

    return clean_kwargs


def _collapse_val(val):
    if (
        (isinstance(val, list) or isinstance(val, tuple))
        and isinstance(val[0], int)
        and val == list(range(val[0], val[-1] + 1))
    ):
        return f"range({val[0]}, {val[-1]+1})"
    elif (isinstance(val, list) or isinstance(val, tuple)) and isinstance(
        val[0], float
    ):
        return str([f"{x:.2e}" for x in val])
    else:
        return str(val)


def dict2string(d):
    """Convert a dictionary to a string representation."""
    if not d:
        return ""
    return "_".join(f"{k}={_collapse_val(v)}" for k, v in sorted(d.items()))
