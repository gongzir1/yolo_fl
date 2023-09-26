import argparse
import flwr as fl
import torch
import csv

from src.client import YOLO_Client
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

"""
ALERT! Check that yolov5s.pt is updating correctly. maybe the best.pt of each run is the one to use.
"""


def main(opt) -> None:
    """Create and start YOLO_Client."""
    # Configure logger
    fl.common.logger.configure(f"client_{opt.cid}", host=opt.log_host)

    # Start client
    client = YOLO_Client(opt.cid, opt)
    fl.client.start_client(opt.server_address, client)

    # record memory usage



    # Enable CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start memory recording
    # torch.cuda.reset_max_memory_allocated(device=device)
    # torch.cuda.reset_max_memory_cached(device=device)

    # Perform your operations, training, or inference
    # ...

    # # Record memory usage after the operations
    # max_memory_allocated = torch.cuda.max_memory_allocated(device=device)
    # max_memory_cached = torch.cuda.max_memory_cached(device=device)
    #
    # print(f"Peak memory allocated: {max_memory_allocated / 1024**2:.2f} MB")
    # print(f"Peak memory cached: {max_memory_cached / 1024**2:.2f} MB")
    #
    # # Save memory usage to a CSV file
    # csv_filename = "memory_usage.csv"
    # with open(csv_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Peak Memory Allocated (MB)", "Peak Memory Cached (MB)"])
    #     writer.writerow([max_memory_allocated / 1024**2, max_memory_cached / 1024**2])
    #
    # print(f"Memory usage data saved to {csv_filename}.")


# def main(opt) -> None:
#     """Create and start YOLO_Client."""
#     # Configure logger
#     fl.common.logger.configure(f"client_{opt.cid}", host=opt.log_host)
#
#     # List of client IDs and options
#     client_list = []
#
#     for i in range(1, 51):
#         client_opt = {
#             'cid': i,
#             'server_address': f'server{i}_address',
#             'log_host': f'log_host{i}'
#         }
#         client_list.append(client_opt)
#
#     # Loop over the list of clients and start each one
#     for client_opt in client_list:
#         # Start client
#         client = YOLO_Client(client_opt['cid'], client_opt)
#         fl.client.start_client(client_opt['server_address'], client)




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
        "--cid", type=str, required=False, help="Client CID (no default)"

    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path"
    )
    parser.add_argument(
        "--cfg", type=str, default="models/yolov5s.yaml", help="model.yaml path"
    )
    parser.add_argument(
        "--data", type=str, default=ROOT / "data/coco.yaml", help="dataset.yaml path"
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default=ROOT / "data/hyps/hyp.scratch.yaml",
        help="hyperparameters path",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="total batch size for all GPUs"
    )
    parser.add_argument('--mode', type=str, default='fedbn', help='fedavg | fedprox | fedbn')
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
        "--workers", type=int, default=10 , help="maximum number of dataloader workers"
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
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

