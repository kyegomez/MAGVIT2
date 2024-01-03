import csv
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from decord import VideoReader
from torch.utils.data.dataset import Dataset


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0):
        print("### " + s)


class WebVid10M(Dataset):
    def __init__(
        self,
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        is_image=False,
    ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        self.pixel_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(sample_size[0]),
                transforms.CenterCrop(sample_size),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = (
            video_dict["videoid"],
            video_dict["name"],
            video_dict["page_dir"],
        )

        video_dir = os.path.join(self.video_folder, page_dir, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        if not self.is_image:
            clip_length = min(
                video_length, (self.sample_n_frames - 1) * self.sample_stride + 1
            )
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(
                start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
            )
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = (
            torch.from_numpy(video_reader.get_batch(batch_index).asnumpy())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pixel_values = pixel_values / 255.0
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception:
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample
