"""Minimal example on how to start a simple Flower server."""

import argparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from src.utils import creation_of_the_model, eval_net
from src.strategy import FedAvg_c
from src.server import Server


from src.client import get_weights

from pathlib import Path
import sys
import os
import time
import yaml
# pylint: disable=no-member
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import check_file

import yaml

# def update_yaml_train_path(yaml_file, variable1, variable2):
#     # client_number = {
#     #     2: 'data/datasets/meat_2/iid/client_1/train',
#     #     3: 'data/datasets/meat_3/iid/client_1/train',
#     #     4: 'data/datasets/meat_4/ex_non_iid/client3',
#     #     5: 'data/datasets/meat_5/iid/client_1/train',
#     #     6: 'data/datasets/meat_6/ex_non_iid/client3',
#     #     7: 'data/datasets/meat_7/iid/client_1/train',
#     #     8: 'data/datasets/meat_8/ex_non_iid/client3',
#     #     9: 'data/datasets/meat_9/iid/client_1/train',
#     #     10: 'data/datasets/meat_10/ex_non_iid/client3',
#     # }
#     # distribution = {
#     #     iid: 'some/path/for/variable2',
#     #     non - iid: 'another/path/for/variable2',
#     #     # Add more mappings for variable2 as needed
#     # }
#     path={
#         2: 'data/datasets/meat_2/iid/',
#         3: 'data/datasets/meat_3/iid/',
#         4: 'data/datasets/meat_4/iid/',
#         5: 'data/datasets/meat_5/iid/',
#         6: 'data/datasets/meat_6/iid/',
#         7: 'data/datasets/meat_7/iid/',
#         8: 'data/datasets/meat_8/iid/',
#         9: 'data/datasets/meat_9/iid/',
#         10: 'data/datasets/meat_10/iid/',
#     }
#     path_non_iid = {
#         2: 'data/datasets/meat_2/non_iid/',
#         3: 'data/datasets/meat_3/non_iid/',
#         4: 'data/datasets/meat_4/non_iid/',
#         5: 'data/datasets/meat_5/non_iid/',
#         6: 'data/datasets/meat_6/non_iid/',
#         7: 'data/datasets/meat_7/non_iid/',
#         8: 'data/datasets/meat_8/non_iid/',
#         9: 'data/datasets/meat_9/non_iid/',
#         10: 'data/datasets/meat_10/ex_non_iid/',
#     }
#
#     with open(yaml_file, 'r') as file:
#         data = yaml.safe_load(file)
#
#     if (variable1, variable2) in [(i, 'iid') for i in range(2, 11)]:
#         data['train'] = path[variable1]
#     elif (variable1, variable2) in [(i, 'non_iid') for i in range(2, 11)]:
#         data['train'] = path_non_iid[variable1]
#     else:
#         print(
#             f"Invalid input combination for variable1 '{variable1}' and variable2 '{variable2}'. No corresponding train path found.")
#         return
#
#     with open(yaml_file, 'w') as file:
#         yaml.dump(data, file)
#
# def update_all_yaml_files(folder_path, variable1):
#     # for filename in os.listdir(folder_path):
#     #     if filename.endswith(".yaml"):
#     #         yaml_file_path = os.path.join(folder_path, filename)
#     #         update_yaml_train_path(yaml_file_path, f'client{variable1}')
#     for i in range(1, variable1 + 1):
#         client = f'client{i}'
#         for filename in os.listdir(folder_path):
#             if filename.endswith(".yaml"):
#                 yaml_file_path = os.path.join(folder_path, filename)
#                 update_yaml_train_path(yaml_file_path, client)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_address",
        type=str,
        required=False,
        default="[::]:8080",
        help=f"gRPC server address",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of rounds of federated learning (default: 2)",
    )
    parser.add_argument(
        "--load_params",
        dest="load_params",
        action="store_true",
        help="Load existing parameters for selected model",
    )
    parser.add_argument(
        "--no-load_params",
        dest="load_params",
        action="store_false",
        help="Do Not load existing parameters for selected model",
    )
    parser.set_defaults(load_params=True)
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="number of workers for dataset reading",
    )
    parser.add_argument("--pin_memory", action="store_true")

    # YOLO params
    parser.add_argument(
        "--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path"
    )
    parser.add_argument(
        "--cfg", type=str, default="models/yolov5s.yaml", help="model.yaml path"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/voc1.yaml",
        help="dataset.yaml path",
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default=ROOT / "data/hyps/hyp.scratch.yaml",
        help="hyperparameters path",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=640,
        help="train, val image size (pixels)",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument(
        "--noval", action="store_true", help="only validate final epoch"
    )
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable autoanchor check"
    )
    parser.add_argument(
        "--evolve",
        type=int,
        nargs="?",
        const=300,
        help="evolve hyperparameters for x generations",
    )
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help='--cache images in "ram" (default) or "disk"',
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="cuda device, i.e. 0 or 0,1,2,3,4,5,6,7 or cpu"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", help="vary img-size +/- 50%%"
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="train multi-class data as single-class",
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--workers", type=int, default=12, help="maximum number of dataloader workers"
    )
    parser.add_argument(
        "--project", default=ROOT / "runs/train", help="save to project/name"
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--linear-lr", action="store_true", help="linear LR")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="EarlyStopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Number of layers to freeze. backbone=10, all=24",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every x epochs (disabled if < 1)",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )

    # Weights & Biases arguments
    parser.add_argument("--entity", default=None, help="W&B: Entity")
    parser.add_argument(
        "--upload_dataset",
        action="store_true",
        help="W&B: Upload dataset as artifact table",
    )
    parser.add_argument(
        "--bbox_interval",
        type=int,
        default=-1,
        help="W&B: Set bounding-box image logging interval",
    )
    parser.add_argument(
        "--artifact_alias",
        type=str,
        default="latest",
        help="W&B: Version of dataset artifact to use",
    )

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    # in train handler. would be interesting a server_handler
    opt.data = check_file(opt.data)
    return opt


opt = parse_opt()


def main() -> None:
    """Start server and train five rounds."""

    assert (
        opt.min_sample_size <= opt.min_num_clients
    ), f"Num_clients shouldn't be lower than min_sample_size"

    # Configure logger
    fl.common.logger.configure("server", host=opt.log_host)

    # Load evaluation data
    # _, testset = utils.load_cifar(download=True)
    # _, testset = load_fashionmnist()

    # Load global parameters, if chosen or exist
    params = prepare_server(opt)
    # params = None

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = FedAvg_c(
        fraction_fit=opt.sample_fraction,
        min_fit_clients=opt.min_sample_size,
        min_available_clients=opt.min_num_clients,
        eval_fn=get_eval_fn(),
        on_fit_config_fn=fit_config,
        initial_parameters=params,
    )
    server = Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        opt.server_address, server, config={"num_rounds": opt.rounds}
    )




def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(1),
        "batch_size": str(opt.batch_size),
        "num_workers": str(opt.num_workers),
        "pin_memory": str(opt.pin_memory),
    }
    return config


def set_weights(model, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn() -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the passed testset for evaluation."""

        # model = load_model(args.model)
        model = creation_of_the_model(opt)
        set_weights(model, weights)

        # loss, accuracy = eval_net(opt)
        res_eval = eval_net(
            opt
        )  # res_eval = (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

        print()
        print("RESULT EVAL")
        print(
            "res_eval = (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t"
        )
        print(res_eval)
        print()

        # improve and check if it is right
        loss = res_eval[0]
        accuracy = res_eval[3]
        return loss, {"accuracy": accuracy}

    return evaluate


def prepare_server(opt) -> Tuple:
    """returns the model parameters if requested.

    opt: options paesed with parse_opt()
    """
    # device = "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if opt.load_params is True:
        try:
            model = creation_of_the_model(opt)
            params = get_weights(model)

        except FileNotFoundError:
            print("Maybe you wanted the argument --no-load_params")

    else:
        params = None

    return params


if __name__ == "__main__":
    # update_all_yaml_files(folder_path, variable1)
    main()



