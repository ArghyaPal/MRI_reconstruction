import argparse
import pathlib

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", help="The path of the directory to the dataset", required=True)
    parser.add_argument("--mode", help="Choice of space to find the nearest neighbour", choices=['image', 'kspace'], default='image')
    parser.add_argument("--center-fractions", nargs='+', default=[0.08, 0.04])
    parser.add_argument("--accelerations", nargs='+', default=[4, 8])
    parser.add_argument("--resolution", default=320, type=int)
    parser.add_argument("--sample-rate", default=1.)
    parser.add_argument("--challenge", default="singlecoil", choices=["singlecoil", "multicoil"])

    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--learning-rate", default=0.0001, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--reluslope", default=0.2, type=float)

    parser.add_argument("--checkpoint", default='DAE/best_model.pt')
    parser.add_argument("--exp-dir", default='DAE')
    parser.add_argument("--resume", default=False, type=bool, choices=[True, False])
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--non-zero-ratio', type=float, default=0.0, help='Minimum percentage of non-zero values allowed')
    parser.add_argument('--weight-decay', type=float, default=0., help='Strength of weight decay regularization')
    parser.add_argument('--reduce', type=bool, default=True, help='Whether to reduce kspace to size 320x320')
    parser.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path("output"), help='Name of the directory to store the output')

    return parser.parse_args()