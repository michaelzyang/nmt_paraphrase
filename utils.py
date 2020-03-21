import os, sys
import torch


def check_save_dir(save_dir):
    done = False
    while not done:
        #  Make sure save_dir exists
        if not os.path.exists(save_dir):
            print(f"Save directory {save_dir} does not exist.")
            mkdir_save = input("Do you wish to create the directory [m], enter a different directory [n], or exit [e]? ")
            if mkdir_save == 'm':
                os.makedirs(save_dir)
                done = True
                continue
            elif mkdir_save == 'n':
                save_dir = input("Enter new save directory: ")
                continue
            elif mkdir_save == 'e':
                sys.exit()
            else:
                print("Please enter one of [m, n, e]")
                continue

        #  Ensure user knows if save_dir is not empty
        if os.listdir(save_dir):
            use_save_dir = input(f"Save directory {save_dir} is not empty. Are you sure you want to continue? [y/n] ")
            if use_save_dir == 'n':
                save_dir = input("Enter new save directory (or [exit] to exit): ")
            elif use_save_dir == 'y':
                done = True
            else:
                print("Please enter one of [y/n]")

    return save_dir


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


def load_inference(model, checkpoint_path):
    if not checkpoint_path:
        print("In inference mode, a path to a model checkpoint must be provided.")
        checkpoint_path = input(f"Please enter checkpoint path or enter 'exit' to exit: ")

    # Try loading checkpoint and keep asking for valid checkpoint paths upon failure.
    exit_loop = False
    while not exit_loop:
        if checkpoint_path == 'exit':
            sys.exit()
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            exit_loop = True
        except FileNotFoundError:
            print(f"Provided checkpoint path {checkpoint_path} not found. Path must include the filename itself.")
            checkpoint_path = input("Provide a new path, or 'exit' to exit: ")


def init_load_train(model, checkpoint_path, optimizer=None, init_fn=None):
    # Check input
    if not checkpoint_path:
        init = input("Checkpoint path not provided. Do you want to randomly initialize weights [y] or enter a path [n]? ")
        while init not in ['y', 'n']:
            init = input("Please enter [y / n] ")
        if init == 'y':
            checkpoint_path = 'init'
        else:
            checkpoint_path = input("Please input the checkpoint path ")

    # Check load path
    if checkpoint_path != 'init':
        # Try loading checkpoint and keep asking for valid checkpoint paths upon failure.
        # If user enters 'init', abandon loading weights and initialize weights instead.
        exit_loop = False
        while not exit_loop:
            try:
                checkpoint = torch.load(checkpoint_path)
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
                print(f"Error loading checkpoint from {checkpoint_path}. Note that path must include the filename itself.")
                checkpoint_path = input("Provide a new path, or 'init' to randomize weights: ")
                if checkpoint_path == 'init':
                    exit_loop = True

    # Initialize weights if required
    if checkpoint_path == 'init':
        if init_fn:
            model.apply(init_fn)
        start_epoch = 1

    return start_epoch
