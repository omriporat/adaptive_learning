import argparse
import yaml
import os

CONFIG_DIR = "config"
CONFIG_NAME = "config.yaml"

def parse_args(args=None):
    print(args)
    parser = argparse.ArgumentParser(description="Load configuration for adaptive learning.")
    parser.add_argument(
        "--substrate",
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--opmode",
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--no-finetune",
        dest="finetune",
        action="store_false",
        default=argparse.SUPPRESS
    )
    parser.add_argument("--log_for_bootstrap",
        action="store_true",
        default=argparse.SUPPRESS)
    parser.add_argument("--no-log_for_bootstrap",
        dest="log_for_bootstrap",
        action="store_false",
        default=argparse.SUPPRESS)
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=argparse.SUPPRESS
    )
    args = parser.parse_args(args)
    # convert to dict
    args = vars(args)
    return args


def load_config(config_path: str = f"{CONFIG_DIR}/{CONFIG_NAME}", args: list = None, create_dirs: bool = True) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if args is not None and len(args) > 0:
        args_dict = parse_args(args)
        # override config with command line arguments if provided
        config = config | args_dict
        
    # build paths based on root_path and other parameters
    root_path = config["root_path"]
    dataset_name = config["dataset_filename"]
    tag = config["tag"]
    data_dir = config["data_dir_name"]
    pretraining_dir = config["pretraining_dir_name"]
    bootstrap_dir = config["bootstrap_dir_name"]
    weights_file = config["weights_filename"]

    config["dataset_path"] = f"{root_path}/{data_dir}/{config['enzyme']}/{tag}/{dataset_name}"
    config["substrate_specific_dataset_path"] = f"{root_path}/{data_dir}/{config['enzyme']}/{tag}/{config['substrate']}/{dataset_name}"
    config["save_path"] = f"{root_path}/{pretraining_dir}/{config['enzyme']}/{tag}"
    config["bootstrap_path"] = f"{config['save_path']}/{bootstrap_dir}/"
    config["weights_path"] = f"{config['save_path']}/{weights_file}"
    opmode = config["opmode"]
    finetune = config["finetune"]
    finetune_tag = "finetune" if finetune else "naive"
    config["finetune_tag"] = finetune_tag
    config["embeddings_path"] = f"{root_path}/{data_dir}/{config['enzyme']}/{tag}/embeddings_{opmode}_{finetune_tag}.npy"
    config["embeddings_clusters_path"] = f"{config['save_path']}/clusters_{opmode}_{finetune_tag}.npy"
    config["sequence_clusters_path"] = f"{config['save_path']}/sequence_clusters_{opmode}_{finetune_tag}.npy"

    config["data_dir_path"] = f"{root_path}/{data_dir}/{config['enzyme']}/{tag}"
    config["results_path"] = f"{root_path}/{config['results_dir_name']}/{config['enzyme']}/{tag}"

    config["substrate_specific_results_path"] = f"{config['results_path']}/{config['substrate']}"
    config["substrate_specific_dataset_path"] = f"{root_path}/{data_dir}/{config['enzyme']}/{tag}/{config['substrate']}/{dataset_name}"

    config["bootstrap_results_path"] = f"{config['substrate_specific_results_path']}/bootstrap_{opmode}_{finetune_tag}.csv"

    if create_dirs:
        # create directories if they don't exist
        os.makedirs(config["save_path"], exist_ok=True)
        os.makedirs(config["bootstrap_path"], exist_ok=True)
        os.makedirs(os.path.dirname(config["embeddings_path"]), exist_ok=True)
        os.makedirs(config["data_dir_path"], exist_ok=True)
        os.makedirs(config["results_path"], exist_ok=True)
        os.makedirs(config["substrate_specific_results_path"], exist_ok=True)

    # pretty print the config
    print("Loaded configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return config