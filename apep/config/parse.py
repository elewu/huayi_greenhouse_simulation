
import sys



def parse_command_line_arguments():
    arguments = sys.argv[1:]
    args_dict = {}
    for arg in arguments:
        if "=" in arg:
            # Split each argument into a key-value pair at the "=" sign
            key, value = arg.split("=", 1)
            args_dict[key] = value
        else:
            # Arguments without an "=" sign are considered boolean flags
            args_dict[arg] = True
    return args_dict

def args_to_hydra_list(args_dict):
    return [f'{key}={value}' for key, value in args_dict.items()]


