import argparse
from mucoco.decode_new_clean import cli_main, cli_main_arg_file

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--argument_file_path", type=str)
    args = parser.parse_args()
    # cli_main()
    cli_main_arg_file(args.argument_file_path)
