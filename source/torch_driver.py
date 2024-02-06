import argparse
import logging
import sys
import os
import torch

sys.path.append(os.getcwd())
from source.configs import DefaultArgs
from source.utils import Action
from source import torch_acc

logger = logging.getLogger(__name__)


def _parse_general_args(parser, default_args):
    parser.add_argument(
        "--action",
        type=str,
        default=default_args.action,
        choices=default_args.actions,
    )
    parser.add_argument(
        "--save_raw_data_dir",
        type=str,
        default=default_args.save_raw_data_dir,
    )
    parser.add_argument(
        "--save_metadata_dir",
        type=str,
        default=default_args.save_metadata_dir,
    )
    parser.add_argument(
        "--assert_device",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logging_level",
        type=int,
        default=default_args.logging_level,
    )

    args, _ = parser.parse_known_args()

    if args.assert_device:
        assert torch.cuda.is_available(), "cuda devices are not available"

    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("source.torch_acc").setLevel(args.logging_level)
    logging.getLogger("__main__").setLevel(args.logging_level)

    logger.debug("added general args to parser.")
    logger.debug(f"args: {args}")

    return args


def base_parser(parser, default_args: DefaultArgs):
    args = _parse_general_args(parser, default_args)
    if args.action == Action.compute_accuracy_at_q:
        action_args = _parse_accuracy_at_q_args(parser, default_args)
        driver_args = argparse.Namespace(
            action=args.action,
            save_metadata_dir=args.save_metadata_dir,
        )
    else:
        raise NotImplementedError("other actions are not implemented")

    return driver_args, action_args


def _parse_accuracy_at_q_args(parser, default_args):
    parser.add_argument(
        "--q",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--glob_path",
        type=str,
        default="sl_merged_*.csv",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_args.batch_size,
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=default_args.prefetch_factor,
    )
    parser.add_argument(
        "--save_file_name_prefix",
        type=str,
        default="q_",
    )
    args, _ = parser.parse_known_args()

    return args


parser = argparse.ArgumentParser()
driver_args, action_args = base_parser(parser, DefaultArgs)

logger = logging.getLogger(__name__)
logger.setLevel(logger.getEffectiveLevel())


if driver_args.action == Action.compute_accuracy_at_q:
    for q in action_args.q:
        torch_acc.compute_accuracy_at_q(
            driver_args.save_metadata_dir,
            action_args.prefetch_factor,
            action_args.batch_size,
            action_args.save_file_name_prefix,
            q,
            action_args.glob_path,
        )
    torch_acc.write_auxiliary_metadata(
        driver_args.save_metadata_dir,
        action_args.save_file_name_prefix,
        action_args.glob_path,
    )
else:
    raise NotImplementedError
