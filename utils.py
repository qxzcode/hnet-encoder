from argparse import ArgumentParser
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


def parse_cfg[T: BaseModel](cfg_t: type[T]) -> T:
    """
    Parses arguments into the provided configuration.
    `cfg_t` must be a subclass of `BaseModel`.
    """
    assert issubclass(cfg_t, BaseModel)
    parser = ArgumentParser()
    for k, v in cfg_t.model_fields.items():
        arg_name = k.replace("_", "-")
        arg_type = v.annotation
        kwargs: dict[str, Any] = {
            "type": arg_type,
            "help": v.description,
        }

        # Allow choices with literal
        if get_origin(arg_type) == Literal:
            choices = get_args(arg_type)
            kwargs.update({"choices": choices, "type": type(choices[0])})

        # Set default val
        default = v.get_default()
        if default is not PydanticUndefined:
            kwargs.update({"default": default})
        else:
            kwargs.update({"required": True})

        # Allow flags with bool
        if v.annotation is bool:
            kwargs.pop("type")
            kwargs.update({"action": "store_true"})

        parser.add_argument(f"--{arg_name}", **kwargs)
    args = parser.parse_args()
    cfg = cfg_t(**{k.replace("-", "_"): v for k, v in args.__dict__.items()})
    return cfg
