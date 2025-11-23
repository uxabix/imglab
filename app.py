import shlex

from commands import find_command, parse_user_args, print_menu
from pipeline import Pipeline


# ----------------------------------------------------------------------------- #
#                                     MAIN                                      #
# ----------------------------------------------------------------------------- #

def main():
    pipe = Pipeline()
    print("Simple Image Processor CLI")
    print_menu()
    while True:

        try:
            raw = input("> ").strip()
            if not raw:
                continue

            parts = shlex.split(raw)

            op = parts[0]
            tokens = parts[1:]

            cmd = find_command(op)
            if not cmd:
                print("Unknown command. Type 'help' for list.")
                continue

            args = parse_user_args(cmd, tokens)
            cmd["func"](pipe, **args)

        except ValueError as ve:
            print(f"Argument Error: {ve}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()