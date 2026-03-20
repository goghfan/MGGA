import glob
import pickle
import random
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from skimage import transform as skimage_transform
from torch.utils.data import Dataset
from transformers import CLIPTokenizer


def pkload(fname: str):
    with open(fname, "rb") as f:
        return pickle.load(f)


class IXIDataset(Dataset):
    """IXI 配准训练数据集：atlas(fixed) + moving 样本 + 随机器官文本 prompt。"""

    def __init__(
        self,
        data_path: str,
        atlas_path: str,
        img_size: Tuple[int, int, int] = (160, 192, 224),
        medsam_size: int = 1024,
        clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
        max_length: int = 77,
    ):
        self.paths = sorted(glob.glob(data_path))
        self.atlas_path = atlas_path
        self.img_size = img_size
        self.medsam_size = medsam_size
        self.max_length = max_length

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name_or_path)

        self.label_dict = {
            1: "Cerebral White Matter",
            2: "Cerebral Cortex",
            3: "Lateral Ventricle",
            4: "Cerebellum White Matter",
            5: "Cerebellum Cortex",
            6: "Thalamus",
            7: "Caudate",
            8: "Putamen",
            9: "Pallidum",
            10: "Hippocampus",
            11: "Amygdala",
        }

        self.atlas_vol, self.atlas_seg = pkload(self.atlas_path)

    def __len__(self):
        return len(self.paths)

    def preprocess_2d(self, img_2d: np.ndarray) -> torch.Tensor:
        img_resized = skimage_transform.resize(
            img_2d,
            (self.medsam_size, self.medsam_size),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )
        img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)

        # MedSAM image_encoder 期望 3 通道输入
        img_3c = np.repeat(img_resized[np.newaxis, :, :], 3, axis=0)
        return torch.tensor(img_3c).float()

    def tokenize_text(self, text: str) -> torch.Tensor:
        out = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        return out

    def __getitem__(self, index: int) -> Dict[str, Any]:
        fixed_vol, fixed_seg = self.atlas_vol, self.atlas_seg
        moving_vol, moving_seg = pkload(self.paths[index])

        fixed_tensor = torch.from_numpy(fixed_vol).float().unsqueeze(0)  # (1, D, H, W)
        moving_tensor = torch.from_numpy(moving_vol).float().unsqueeze(0)
        fixed_seg_tensor = torch.from_numpy(fixed_seg).long().unsqueeze(0)
        moving_seg_tensor = torch.from_numpy(moving_seg).long().unsqueeze(0)

        slice_id = random.randint(0, self.img_size[0] - 1)
        fixed_slice_tensor = self.preprocess_2d(fixed_vol[slice_id, :, :])
        moving_slice_tensor = self.preprocess_2d(moving_vol[slice_id, :, :])

        label_id = random.randint(1, 11)
        text_prompt = self.label_dict.get(label_id, "Brain")
        tokens = self.tokenize_text(text_prompt)

        return {
            "fixed_image": fixed_tensor,
            "moving_image": moving_tensor,
            "fixed_label": fixed_seg_tensor,
            "moving_label": moving_seg_tensor,
            "fixed_slice": fixed_slice_tensor,
            "moving_slice": moving_slice_tensor,
            "tokens": tokens,
            "slice_id": slice_id,
            "text_label_id": label_id,
        }

