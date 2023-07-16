import argparse
import torchvision


def parse_args():
    parser = argparse.ArgumentParser(prog='Download VOC detection')
    parser.add_argument(
        '--subset', type=str,
    )
    parser.add_argument(
        '--dir', type=str,
    )
    return parser.parse_args()


def download(args):
    torchvision.datasets.VOCDetection(
        root=args.dir,
        year='2012',
        image_set=args.subset,
        download=True,
    )


if __name__ == '__main__':
    download(parse_args())
