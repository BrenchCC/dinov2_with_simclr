import sys
import os
import logging
from enum import Enum
from typing import Union, Callable, Optional
import numpy as np


from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")
_Target = int

class BrenchDataResourceDataset(ExtendedVisionDataset):
    """
    A dataset class for a custom data resource, structured similarly to ImageNet.
    This dataset is assumed to have no specific labels, so a dummy label is used for all images.
    """

    Target = Union[_Target]

    class Split(Enum):
        TRAIN = "train"
        VAL = "val"

        def get_dirname(self):
            return self.value

        @property
        def length(self):
            images_dir = os.path.join(self.root, self.get_dirname(), "images")
            return len([f for f in os.listdir(images_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    def __init__(
            self,
            *,
            root: str,
            split: "BrenchDataResourceDataset.Split",
            extra: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_name = None

    def __len__(self) -> int:
        return len(self._get_entries())

    @property
    def split(self) -> "BrenchDataResourceDataset.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode = "r")

    def _save_extra(self, extra: np.ndarray, extra_path: str, ) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(os.path.dirname(extra_full_path), exist_ok = True)
        np.save(extra_full_path, extra)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"classes_{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"classes_{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        return self._class_ids

    def _get_class_names(self) -> str:
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        return self._class_names

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        image_relpath = entries[index]["image_relpath"]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode = "rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Optional[_Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return int(class_index)

    def get_class_id(self, index: int) -> str:
        class_ids = self._get_class_ids()
        class_index = self.get_target(index)
        return str(class_ids[class_index])

    def get_class_name(self, index: int) -> str:
        class_names = self._get_class_names()
        class_index = self.get_target(index)
        return str(class_names[class_index])

    def _dump_entries(self) -> None:
        """
        collects relative path of all images, save to entries-<SPLIT>.npy
        """
        images_dir = os.path.join(
            self._root,
            self.split.get_dirname(),
            "images",
        )

        # Store the same format with origin; related to root: relative path
        rel_paths = []
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rel = os.path.join(self.split.get_dirname(), "images", fname)
            rel_paths.append(rel)

        # Since there are no real labels, just assign a dummy class_index 0 for all samples
        dtype = np.dtype(
            [
                ("image_relpath", "U256"),
                ("class_index", "<u4"),
            ]
        )
        entries_array = np.empty(len(rel_paths), dtype = dtype)
        for i, rel_path in enumerate(rel_paths):
            entries_array[i] = (rel_path, 0)

        logger.info(f'Saving {len(entries_array)} entries to "{self._entires_path}"')
        self._save_extra(entries_array, self._entires_path)

    def _dump_class_ids_and_names(self) -> None:
        """
        Since there are no labels, only using '0' and a placeholder name
        """
        # only have one class with index 0
        class_ids_array = np.array(["0"], dtype = "U1")
        class_names_array = np.array(["no_label"], dtype = "U10")

        logger.info(f"Saving class IDs to '{self._class_ids_path}'")
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f"Saving class names to '{self._class_names_path}'")
        self._save_extra(class_names_array, self._class_names_path)


    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_calss_ids_and_names()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m dinov2.data.datasets.brench_data_resource /path/to/your/data")
        sys.exit(1)

    root = sys.argv[1]
    extra_dir = os.path.join(root, "extra")

    for split in (BrenchDataResourceDataset.Split.TRAIN, BrenchDataResourceDataset.Split.VAL):
        logger.info(f"Dumping extra files for {split.value} split ...")
        dataset = BrenchDataResourceDataset(root = root, split = split, extra = extra_dir)
        dataset.dump_extra()
    print("✅ all extra files are ready in <root>/extra")
