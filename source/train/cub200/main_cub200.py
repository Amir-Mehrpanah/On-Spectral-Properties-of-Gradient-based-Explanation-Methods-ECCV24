import argparse
import os
import jax
import train_cub200 as train

# https://github.com/google-research/google-research/blob/master/interpretability_benchmark/train_resnet.py#L126
cub_200_params = {
    "train_batch_size": 256,
    "num_train_images": 5994,
    "num_eval_images": 5794,
    "num_label_classes": 200,
    "num_train_steps": 20000,
    "base_learning_rate": 0.1,
    "weight_decay": 0.0001,
    "eval_batch_size": 256,
    "num_channels": 3,
    # from https://github.com/chou141253/FGVC-HERBS/blob/master/data/dataset.py
    "mean_rgb": [0.485, 0.456, 0.406],
    "stddev_rgb": [0.229, 0.224, 0.225],
}


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
        "--work_dir",
        type=str,
        default="source/train/cub200/",
        help="Directory for logging and checkpoints.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/local",
        help="Directory for storing data.",
    )
    parser.add_argument(
        "--name", type=str, default="test", help="Name of this experiment."
    )
    parser.add_argument(
        "--group", type=str, default="default", help="Group name of this experiment."
    )
    # Training
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="Architecture.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from best checkpoint."
    )
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=cub_200_params["base_learning_rate"],
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Number of warmup epochs with lower learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=cub_200_params["train_batch_size"],
        help="Batch size.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=cub_200_params["num_label_classes"],
        help="Number of classes.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=cub_200_params["weight_decay"],
        help="Weight decay.",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for optimizer."
    )
    parser.add_argument("--img_size", type=int, default=224, help="Image size.")
    parser.add_argument(
        "--img_channels",
        type=int,
        default=cub_200_params["num_channels"],
        help="Number of image channels.",
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Use mixed precision training."
    )
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--mean_rgb",
        type=float,
        default=cub_200_params["mean_rgb"],
        help="Mean to be removed from images.",
    )
    parser.add_argument(
        "--stddev_rgb",
        type=float,
        default=cub_200_params["stddev_rgb"],
        help="Divide images by STD.",
    )

    # Logging
    parser.add_argument(
        "--log_every",
        type=int,
        default=cub_200_params["num_train_images"]
        // cub_200_params["train_batch_size"],
        help="Log every log_every steps.",
    )
    args = parser.parse_args(raw_args)

    if jax.process_index() == 0:
        args.ckpt_dir = os.path.join(
            args.work_dir, "runs"
        )
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    train.train_and_evaluate(args)


if __name__ == "__main__":
    main()
