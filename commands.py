import sys
import Filters as F
import ArithmeticalOperations as A

def parse_angles(values):
    if not values:
        return ("0", "90")
    return tuple(values)

COMMANDS = {
    "global": {
        "load": {
            "id": 1,
            "name": "load",
            "aliases": ["1"],
            "args": [
                {"name": "filename", "type": str, "required": True}
            ],
            "desc": "Load an image from file.",
            "func": lambda pipe, filename: pipe.load(filename),
        },
        "save": {
            "id": 2,
            "name": "save",
            "aliases": ["2"],
            "args": [
                {"name": "filename", "type": str, "required": True}
            ],
            "desc": "Save the current result to file.",
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
            "func": lambda pipe, size: pipe.apply(F.mean_filter, size)
        },
        "sharpen": {
            "id": 7,
            "name": "sharpen",
            "aliases": ["7"],
            "args": [],
            "desc": "Sharpen filter.",
            "func": lambda pipe: pipe.apply(F.sharpen_filter)
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
            "func": lambda pipe, sigma, size: pipe.apply(F.gaussian_filter, sigma, size)
        },
        "median": {
            "id": 9,
            "name": "median",
            "aliases": ["9"],
            "args": [
                {"name": "ksize", "type": int, "required": False, "default": 3}
            ],
            "desc": "Median filter.",
            "func": lambda pipe, ksize: pipe.apply(F.median_filter, ksize)
        },
        "sobel": {
            "id": 10,
            "name": "sobel",
            "aliases": ["10"],
            "args": [
                {"name": "angles", "type": parse_angles, "required": False, "default": ("0", "90")}
            ],
            "desc": "Sobel edge detector.",
            "func": lambda pipe, angles: pipe.apply(F.sobel, angles)
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
            "func": lambda pipe, value: pipe.apply(A.image_add, value)
        },
        "sub": {
            "id": 12,
            "name": "sub",
            "aliases": ["12"],
            "args": [
                {"name": "value", "type": int, "required": True}
            ],
            "desc": "Subtract value from all pixels.",
            "func": lambda pipe, value: pipe.apply(A.image_subtract, value)
        },
        "mul": {
            "id": 13,
            "name": "mul",
            "aliases": ["13"],
            "args": [
                {"name": "value", "type": float, "required": True}
            ],
            "desc": "Multiply image by value.",
            "func": lambda pipe, value: pipe.apply(A.image_multiply, value)
        },
        "div": {
            "id": 14,
            "name": "div",
            "aliases": ["14"],
            "args": [
                {"name": "value", "type": float, "required": True}
            ],
            "desc": "Divide image by value.",
            "func": lambda pipe, value: pipe.apply(A.image_divide, value)
        },
        "gamma": {
            "id": 15,
            "name": "gamma",
            "aliases": ["15"],
            "args": [
                {"name": "value", "type": float, "required": True}
            ],
            "desc": "Gamma correction.",
            "func": lambda pipe, value: pipe.apply(A.gamma_correction, value)
        },
        "gray": {
            "id": 16,
            "name": "gray",
            "aliases": ["16"],
            "args": [
                {"name": "method", "type": str, "required": True}
            ],
            "desc": "Convert to grayscale.",
            "func": lambda pipe, method: pipe.apply(A.gray_scale, method)
        },
    }
}

def find_command(name):
    """Search command by name or ID across all registries."""
    for group in COMMANDS.values():
        for cmd in group.values():
            if name == cmd["name"] or name in cmd["aliases"]:
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

    # Case 2: user passed positional-only for required args
    pos_values = [t for t in tokens if "=" not in t]
    pos_index = 0

    for spec in cmd["args"]:
        if spec["name"] not in args:
            if spec["required"]:
                if pos_index >= len(pos_values):
                    raise ValueError(f"Missing required argument: {spec['name']}")
                args[spec["name"]] = spec["type"](pos_values[pos_index])
                pos_index += 1
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