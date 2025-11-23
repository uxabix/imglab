from pipeline import Pipeline
from commands import find_command, parse_user_args, print_menu


# ----------------------------------------------------------------------------- #
#                                     MAIN                                      #
# ----------------------------------------------------------------------------- #

def main():
    pipe = Pipeline()
    print("Simple Image Processor CLI")

    while True:
        print_menu()

        raw = input("> ").strip()
        if not raw:
            continue

        parts = raw.split()
        op = parts[0]
        tokens = parts[1:]

        try:
            cmd = find_command(op)
            if not cmd:
                print("Unknown command. Type 'help' for list.")
                continue

            args = parse_user_args(cmd, tokens)
            cmd["func"](pipe, **args)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()