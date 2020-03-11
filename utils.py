import os, sys
import torch


def check_save_path(save_path):
    if not os.path.exists(save_path):
        while not os.path.exists(save_path):
            print(f"Save path {save_path} does not exist.")
            mkdir_save = input("Do you wish to create the directory [m], enter a different directory [n], or exit [e]? ")
            if mkdir_save == 'm':
                os.makedirs(save_path)
            elif mkdir_save == 'n':
                save_path = input("Enter new save path directory: ")
            elif mkdir_save == 'e':
                sys.exit()
            else:
                mkdir_save = input("Please enter one of [m, n, e]: ")
    return save_path


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        use_cpu = input("Torch device is set to 'cpu'. Are you sure you want to continue? [y/n] ")
        while use_cpu not in ('y', 'n'):
            print("Please input y or n")
            use_cpu = input("Torch device is set to 'cpu'. Are you sure you want to continue? [y/n] ")
        if use_cpu == 'n':
            sys.exit()
    return device


def load_inference(model, load_path):
    if not load_path:
        print("In inference mode, load path to a model checkpoint must be provided.")
        load_path = input(f"Please enter load path or enter 'exit' to exit: ")

    # Try loading model and keep asking for valid load paths upon failure.
    exit_loop = False
    while not exit_loop:
        if load_path == 'exit':
            sys.exit()
        try:
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            exit_loop = True
        except FileNotFoundError:
            print(f"Provided load path {load_path} not found. Path must include the filename itself.")
            load_path = input("Provide a new path, or 'exit' to exit: ")


def init_load_train(model, load_path, optimizer=None, init_fn=None):
    # Check input
    if not load_path:
        init = input("Load path not provided. Do you want to randomly initialize weights [y] or enter a path [n]? ")
        while init not in ['y', 'n']:
            init = input("Please enter y or n ")
        if init == 'y':
            load_path = 'init'
        else:
            load_path = input("Please input the checkpoint path ")

    # Check load path
    if load_path != 'init':
        # Try loading model and keep asking for valid load paths upon failure.
        # If user enters 'init', abandon loading weights and initialize weights instead.
        exit_loop = False
        while not exit_loop:
            try:
                checkpoint = torch.load(load_path)
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    no_optimizer = input("Optimizer not provided. Are you sure you want to continue? [y/n] ")
                    while no_optimizer not in ['y', 'n']:
                        no_optimizer = input("Please enter y or n ")
                    if no_optimizer == 'n':
                        print("Exit script")
                        sys.exit()
                exit_loop = True
            except FileNotFoundError:
                print(f"Provided load path {load_path} not found. Path must include the filename itself.")
                load_path = input("Provide a new path, or 'init' to randomize: ")
                if load_path == 'init':
                    exit_loop = True

    # Initialize weights if required
    if load_path == 'init':
        model.apply(init_fn)
        start_epoch = 1

    return start_epoch
