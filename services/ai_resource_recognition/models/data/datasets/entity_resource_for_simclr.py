import os
import io
from enum import Enum
import logging
from typing import Callable, Optional, Union

import numpy as np
from PIL import Image
import PIL
from pyarrow.fs import FileSystem
from torchvision.datasets import VisionDataset

from ai_resource_recognition.models.data.transforms import get_simclr_pipeline_transform, ContrastiveLearningViewGenerator


logger = logging.getLogger("dinov2")
_Target = int


class EntityResourceForSimCLR(VisionDataset):
    """
    A dataset class for a custom resource, structured similarly to ImageNet.
    This dataset is assumed to have no specific labels, so a dummy label is used for all images.
    """

    Target = Union[_Target]

    class Split(Enum):
        TRAIN = "train"
        VAL = "val"

        def get_dirname(self):
            return self.value

    def __init__(
        self,
        root: str,
        split: "EntityResourceForSimCLR.Split",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        hdfs_load: bool = False,
        crop_size=32,
        n_views=2,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.filenames = []
        self._split = split
        with open(os.path.join(root, f"{split.value.upper()}.txt"), "r") as f:
            self.filenames = [l.strip() for l in f]

        if hdfs_load:
            self._hdfs_client, _ = FileSystem.from_uri('hdfs://haruna/home/')
        else:
            self._hdfs_client = None
        
        self.transform = ContrastiveLearningViewGenerator(
            base_transform=get_simclr_pipeline_transform(crop_size),
            n_views=n_views
        )

    def __getitem__(self, index: int):
        filepath = self.filenames[index]
        if self._hdfs_client:
            with self._hdfs_client.open_input_file(filepath) as f:
                image_data = f.read()
        else:
            with open(filepath, mode="rb") as f:
                image_data = f.read()
        try:
            image = Image.open(io.BytesIO(image_data))
        except PIL.UnidentifiedImageError as p:
            print(f"Error open {index}: {filepath}")
            new_index = min(self.__len__() - 1, index + 1)
            return self.__getitem__(new_index)
        
        transform_images = self.transform(image)
        return transform_images

    def __len__(self) -> int:
        return len(self.filenames)

    @property
    def split(self) -> "EntityResourceForSimCLR.Split":
        return self._split


if __name__ == "__main__":
    # Example: Generate extra files from the script
    # python -m dinov2.data.datasets.entity_resource /path/to/your/data
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m dinov2/data/datasets/entity_resource_for_simclr /path/to/your/data true/false")
        sys.exit(1)

    root = sys.argv[1]
    hdfs_loading_str = sys.argv[2]
    extra_dir = os.path.join(root, "extra")
    hdfs_loading = True if hdfs_loading_str.lower() == "true" else False
    split = "train"
    split = EntityResourceForSimCLR.Split.TRAIN if split == "train" else EntityResourceForSimCLR.Split.VAL

    train_dataset = EntityResourceForSimCLR(root, split, hdfs_load=hdfs_loading)
    num_dataset = len(train_dataset)
    sample = train_dataset[0]
    print(sample[0].shape)
    print("dataset len: ", len(train_dataset))
