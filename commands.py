import sys

import ArithmeticalOperations as A
import Filters as F
import MorphologicalFilters as M
from config import cfg


def parse_angles(val_str):
    """Parses a string like '0,90' or '(0, 90)' into a tuple of strings."""
    if not val_str:
        return ("0", "90")

    cleaned = val_str.replace("(", "").replace(")", "").replace("'", "").replace('"', "")

    if not cleaned:
        return ("0", "90")

    if "," in cleaned:
        return tuple(x.strip() for x in cleaned.split(",") if x.strip())

    return (cleaned,)


def parse_se_size(val_str):
    """Parse structuring element size. Returns integer size (e.g., 3 for 3x3)."""
    if not val_str:
        return 3
    return int(val_str)


COMMANDS = {
    "global": {
        "load": {
            "id": 1,
            "name": "load",
            "aliases": ["1"],
            "args": [
                {"name": "filename", "type": str, "required": True}
            ],
            "desc": "Load an image from file (relative to input_dir).",
            "func": lambda pipe, filename: pipe.load(filename),
        },
        "save": {
            "id": 2,
            "name": "save",
            "aliases": ["2"],
            "args": [
                {"name": "filename", "type": str, "required": True}
            ],
            "desc": "Save the current result (relative to output_dir).",
            "func": lambda pipe, filename: pipe.save(filename),
        },
        "undo": {
            "id": 3,
            "name": "undo",
            "aliases": ["3"],
            "args": [],
            "desc": "Undo last operation.",
            "func": lambda pipe: pipe.undo()
        },
        "redo": {
            "id": 4,
            "name": "redo",
            "aliases": ["4"],
            "args": [],
            "desc": "Redo previously undone operation.",
            "func": lambda pipe: pipe.redo()
        },
        "exit": {
            "id": 5,
            "name": "exit",
            "aliases": ["5"],
            "args": [],
            "desc": "Exit the program.",
            "func": lambda pipe: sys.exit(0)
        },
        "history": {
            "id": 99,
            "name": "history",
            "aliases": ["h"],
            "args": [],
            "desc": "Show execution history.",
            "func": lambda pipe: pipe.print_history()
        }
    },

    "configuration": {
        "set_input": {
            "id": 20,
            "name": "set_input",
            "aliases": ["inp"],
            "args": [{"name": "path", "type": str, "required": True}],
            "desc": "Set input directory.",
            "func": lambda pipe, path: (cfg.set("input_dir", path), print(f"Input dir set to: {path}"))
        },
        "set_output": {
            "id": 21,
            "name": "set_output",
            "aliases": ["out"],
            "args": [{"name": "path", "type": str, "required": True}],
            "desc": "Set output directory.",
            "func": lambda pipe, path: (cfg.set("output_dir", path), print(f"Output dir set to: {path}"))
        },
        "mode": {
            "id": 22,
            "name": "mode",
            "aliases": ["m"],
            "args": [
                {"name": "val", "type": int, "required": True},
                {"name": "filename", "type": str, "required": False, "default": "output.png"}
            ],
            "desc": "Set mode: 0=Manual, 1=Auto-NewFile, 2=Auto-FixedFile.",
            "func": lambda pipe, val, filename: (
                cfg.set("mode", val),
                cfg.set("fixed_filename", filename),
                print(f"Mode set to {val}" + (f" (file: {filename})" if val == 2 else ""))
            )
        },
        "info": {
            "id": 23,
            "name": "info",
            "aliases": ["i"],
            "args": [],
            "desc": "Show current configuration.",
            "func": lambda pipe: print(f"Config: {cfg.config}")
        }
    },

    "filters": {
        "mean": {
            "id": 6,
            "name": "mean",
            "aliases": ["6"],
            "args": [
                {"name": "size", "type": int, "required": False, "default": 3}
            ],
            "desc": "Mean (box) filter.",
            "func": lambda pipe, size: pipe.apply(F.mean_filter, "mean", size)
        },
        "sharpen": {
            "id": 7,
            "name": "sharpen",
            "aliases": ["7"],
            "args": [],
            "desc": "Sharpen filter.",
            "func": lambda pipe: pipe.apply(F.sharpen_filter, "sharpen")
        },
        "gaussian": {
            "id": 8,
            "name": "gaussian",
            "aliases": ["8"],
            "args": [
                {"name": "sigma", "type": float, "required": False, "default": 1.0},
                {"name": "size", "type": int, "required": False, "default": 3},
            ],
            "desc": "Gaussian blur.",
            "func": lambda pipe, sigma, size: pipe.apply(F.gaussian_filter, "gaussian", sigma, size)
        },
        "median": {
            "id": 9,
            "name": "median",
            "aliases": ["9"],
            "args": [
                {"name": "ksize", "type": int, "required": False, "default": 3}
            ],
            "desc": "Median filter.",
            "func": lambda pipe, ksize: pipe.apply(F.median_filter, "median", ksize)
        },
        "sobel": {
            "id": 10,
            "name": "sobel",
            "aliases": ["10"],
            "args": [
                {"name": "angles", "type": parse_angles, "required": False, "default": ("0", "90")}
            ],
            "desc": "Sobel edge detector.",
            "func": lambda pipe, angles: pipe.apply(F.sobel, "sobel", angles)
        },
    },

    "arith": {
        "add": {
            "id": 11,
            "name": "add",
            "aliases": ["11"],
            "args": [
                {"name": "value", "type": int, "required": True}
            ],
            "desc": "Add value to all pixels.",
            "func": lambda pipe, value: pipe.apply(A.image_add, "add", value)
        },
        "sub": {
            "id": 12,
            "name": "sub",
            "aliases": ["12"],
            "args": [
                {"name": "value", "type": int, "required": True}
            ],
            "desc": "Subtract value from all pixels.",
            "func": lambda pipe, value: pipe.apply(A.image_subtract, "sub", value)
        },
        "mul": {
            "id": 13,
            "name": "mul",
            "aliases": ["13"],
            "args": [
                {"name": "value", "type": float, "required": True}
            ],
            "desc": "Multiply image by value.",
            "func": lambda pipe, value: pipe.apply(A.image_multiply, "mul", value)
        },
        "div": {
            "id": 14,
            "name": "div",
            "aliases": ["14"],
            "args": [
                {"name": "value", "type": float, "required": True}
            ],
            "desc": "Divide image by value.",
            "func": lambda pipe, value: pipe.apply(A.image_divide, "div", value)
        },
        "gamma": {
            "id": 15,
            "name": "gamma",
            "aliases": ["15"],
            "args": [
                {"name": "value", "type": float, "required": True}
            ],
            "desc": "Gamma correction.",
            "func": lambda pipe, value: pipe.apply(A.gamma_correction, "gamma", value)
        },
        "gray": {
            "id": 16,
            "name": "gray",
            "aliases": ["16"],
            "args": [
                {"name": "method", "type": str, "required": True}
            ],
            "desc": "Convert to grayscale.",
            "func": lambda pipe, method: pipe.apply(A.gray_scale, "gray", method)
        },
    },

    "morphological": {
        "binary": {
            "id": 17,
            "name": "binary",
            "aliases": ["17"],
            "args": [
                {"name": "threshold", "type": int, "required": False, "default": 128}
            ],
            "desc": "Convert image to binary (0 or 255).",
            "func": lambda pipe, threshold: pipe.apply(M.binary, "binary", threshold)
        },
        "dilate": {
            "id": 18,
            "name": "dilate",
            "aliases": ["18"],
            "args": [
                {"name": "size", "type": parse_se_size, "required": False, "default": 3}
            ],
            "desc": "Morphological dilation (SE: size x size).",
            "func": lambda pipe, size: pipe.apply(
                lambda img: M.dilation(img, np.ones((size, size), dtype=np.uint8)),
                "dilate"
            )
        },
        "erode": {
            "id": 19,
            "name": "erode",
            "aliases": ["19"],
            "args": [
                {"name": "size", "type": parse_se_size, "required": False, "default": 3}
            ],
            "desc": "Morphological erosion (SE: size x size).",
            "func": lambda pipe, size: pipe.apply(
                lambda img: M.erosion(img, np.ones((size, size), dtype=np.uint8)),
                "erode"
            )
        },
        "open": {
            "id": 24,
            "name": "open",
            "aliases": ["24"],
            "args": [
                {"name": "size", "type": parse_se_size, "required": False, "default": 3}
            ],
            "desc": "Morphological opening (erosion → dilation).",
            "func": lambda pipe, size: pipe.apply(
                lambda img: M.opening(img, np.ones((size, size), dtype=np.uint8)),
                "open"
            )
        },
        "close": {
            "id": 25,
            "name": "close",
            "aliases": ["25"],
            "args": [
                {"name": "size", "type": parse_se_size, "required": False, "default": 3}
            ],
            "desc": "Morphological closing (dilation → erosion).",
            "func": lambda pipe, size: pipe.apply(
                lambda img: M.closing(img, np.ones((size, size), dtype=np.uint8)),
                "close"
            )
        },
    }
}

import numpy as np


def find_command(name):
    """Search command by name, alias OR ID string."""
    for group in COMMANDS.values():
        for cmd in group.values():
            if (name == cmd["name"] or
                    name in cmd["aliases"] or
                    name == str(cmd["id"])):
                return cmd
    return None

def parse_user_args(cmd, tokens):
    """Parse key=value arguments and fill missing with defaults."""
    args = {}

    # Case 1: user used explicit key=value
    for token in tokens:
        if "=" in token:
            key, val = token.split("=", 1)

            for spec in cmd["args"]:
                if spec["name"] == key:
                    args[key] = spec["type"](val)
                    break

    # Case 2: user passed positional-only for remaining args
    pos_values = [t for t in tokens if "=" not in t]
    pos_index = 0

    for spec in cmd["args"]:
        if spec["name"] in args:
            continue

        if pos_index < len(pos_values):
            args[spec["name"]] = spec["type"](pos_values[pos_index])
            pos_index += 1
        else:
            if spec["required"]:
                raise ValueError(f"Missing required argument: {spec['name']}")
            else:
                args[spec["name"]] = spec["default"]

    return args

def print_menu():
    print("\nAvailable commands:\n")

    for category_name, group in COMMANDS.items():
        print(f"== {category_name.upper()} ==")
        for cmd in group.values():
            print(f"  {cmd['id']:>2}. {cmd['name']:<10} - {cmd['desc']}")
        print()