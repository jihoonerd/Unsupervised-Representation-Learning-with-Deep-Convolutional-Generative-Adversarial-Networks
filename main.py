import argparse

from dcgan.trainer.trainer import DCGANTrainer


def args_parse():
    parser = argparse.ArgumentParser(description="DCGAN Trainer")
    parser.add_argument(
        "--data",
        default="cifar10",
        help="Dataset name, currently only supports keras dataset",
    )
    parser.add_argument("--keras_dataset", action="store_true", help="Boolean")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=64)
    args, extras = parser.parse_known_args()
    return args, extras


if __name__ == "__main__":
    args, extras = args_parse()
    trainer = DCGANTrainer(*extras, **vars(args))
    trainer.train()
