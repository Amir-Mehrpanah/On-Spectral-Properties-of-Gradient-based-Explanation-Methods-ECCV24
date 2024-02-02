from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import numpy as np
import pandas as pd
import os
import logging
import torch
import sys

sys.path.append(os.getcwd())
from source.project_manager import load_experiment_metadata

logger = logging.getLogger(__name__)

preprocess = torchvision.transforms.Compose(
    [
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


class SLQDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sl_metadata, remove_q=0, verbose=False):
        """
        Arguments:
            sl_metadata (string): Path to the csv metadata file.
            q (float): The quantile value for the saliency mask to remove from the image.
        """
        self.sl_metadata = sl_metadata
        self.q = 100 - remove_q
        self.verbose = verbose

    def __len__(self):
        return len(self.sl_metadata)

    def __getitem__(self, idx):
        original_image_path = self.sl_metadata.iloc[idx]["image_path"]
        image_index = self.sl_metadata.iloc[idx]["image_index"]
        saliency_image_path = self.sl_metadata.iloc[idx]["data_path"]
        label = self.sl_metadata.iloc[idx]["label"]
        alpha_mask_value = self.sl_metadata.iloc[idx]["alpha_mask_value"]

        original_image = Image.open(original_image_path).convert("RGB")
        original_image = preprocess(original_image)

        if self.q < 100 or self.verbose:
            saliency_image = np.load(saliency_image_path)
            saliency_image = torch.tensor(saliency_image)
            # (1, H, W, C) -> (1, H, W)
            saliency_image = torch.sum(
                saliency_image,
                axis=-1,
            )
            mask = saliency_image < np.percentile(saliency_image, self.q)
            masked_image = original_image * mask
        else:
            masked_image = original_image

        if self.verbose:
            sample = {
                "original_image": original_image,
                "saliency": saliency_image,
                "label": label,
                "mask": 1.0 * mask,
                "masked_image": masked_image,
                "image_index": image_index,
                "alpha_mask_value": alpha_mask_value,
            }
        else:
            sample = {
                "masked_image": masked_image,
                "label": label,
            }
        return sample


def compute_accuracy_at_q(
    save_metadata_dir,
    prefetch_factor,
    batch_size,
    save_file_name_prefix,
    q,
    glob_path,
):
    sl_metadata = load_experiment_metadata(save_metadata_dir, glob_path=glob_path)
    slqds = SLQDataset(sl_metadata, remove_q=q)
    slqdl = DataLoader(
        slqds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    forward = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    )
    forward.eval()
    preds = []
    with torch.no_grad():
        for batch in slqdl:
            logits = forward(batch["masked_image"])
            logits = logits.argmax(axis=1)
            preds.append(logits == batch["label"])

    # convert preds to dataframe
    preds = pd.DataFrame(
        {
            "preds": np.concatenate(preds, axis=0),
        },
    )
    preds["q"] = q

    sl_metadata = pd.concat([sl_metadata, preds], axis=1)
    file_name = f"{save_file_name_prefix}_{q}.csv"
    sl_metadata.to_csv(os.path.join(save_metadata_dir, file_name), index=False)
