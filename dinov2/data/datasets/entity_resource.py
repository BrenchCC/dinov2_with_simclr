import os
import json
from enum import Enum
import logging
from typing import Callable, Optional, Union

import numpy as np
from pyarrow.fs import FileSystem


from dinov2.data.tos_client import TosClient
# from .extended import ExtendedVisionDataset
from dinov2.data.datasets.extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")
_Target = int


class EntityResource(ExtendedVisionDataset):
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
        *,
        split: "EntityResource.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        hdfs_load: bool = False,
        tos_load: bool = False,
        tos_config_path: str = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None
        self._hdfs_load = hdfs_load
        if hdfs_load:
            self._hdfs_client, _ = FileSystem.from_uri('hdfs://haruna/home/')
        else:
            self._hdfs_client = None
        
        if tos_load:
            if not tos_config_path:
                raise ValueError("tos config must be provided")
            with open(tos_config_path, "r") as f:
                tos_config = json.load(f)
            self._tos_client = TosClient(**tos_config)
        else:
            self._tos_client = None

    @property
    def split(self) -> "EntityResource.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        return self._class_names

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        image_relpath = entries[index]["image_relpath"]
        image_full_path = os.path.join(self.root, image_relpath)
        if self._hdfs_client:
            with self._hdfs_client.open_input_file(image_relpath) as f:
                image_data = f.read()
        elif self._tos_client:
            image_dict = json.loads(image_relpath)
            _, image_data = self._tos_client.download_file(image_dict)
        else:
            with open(image_full_path, mode="rb") as f:
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

    def __len__(self) -> int:
        return len(self._get_entries())

    def _dump_entries(self) -> None:
        """Collects relative paths of all images and saves them to entries-<SPLIT>.npy"""
        if self._hdfs_load:
            images_path_fp = os.path.join(self.root, self.split.get_dirname(), "frame_hdfs_path.txt")
            rel_paths = []
            with open(images_path_fp, "r") as f:
                for line in f:
                    rel_paths.append(line.strip())  # rel_path is absolute hdfs path in this case.
        elif self._tos_client:
            images_path_fp = os.path.join(self.root, self.split.get_dirname(), "frame_tos_path.jsonl")
            rel_paths = []
            with open(images_path_fp, "r") as f:
                for line in f:
                    rel_paths.append(line.strip())  # rel_path is absolute tos loading path in this case.
        else:
            images_dir = os.path.join(self.root, self.split.get_dirname(), "images")
            rel_paths = []
            for fname in sorted(os.listdir(images_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Store the relative path with respect to the root directory
                rel = os.path.join(self.split.get_dirname(), "images", fname)
                rel_paths.append(rel)

        # Since there are no real labels, we assign a dummy class_index 0 to all samples.
        dtype = np.dtype(
            [
                ("image_relpath", "U1024"),  # Using a fixed-size string for paths
                ("class_index", "<u4"),
            ]
        )
        entries_array = np.empty(len(rel_paths), dtype=dtype)
        for i, rel_path in enumerate(rel_paths):
            entries_array[i] = (rel_path, 0)

        logger.info(f'saving {len(entries_array)} entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:
        """Since there are no labels, we use 0 and a placeholder name"""
        # We only have one class with index 0
        class_ids_array = np.array(["0"], dtype="U1")
        class_names_array = np.array(["no_label"], dtype="U10")

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_class_ids_and_names()


if __name__ == "__main__":
    # Example: Generate extra files from the script
    # python -m dinov2.data.datasets.entity_resource /path/to/your/data
    import sys

    if len(sys.argv) != 5:
        print("Usage: python -m dinov2.data.datasets.entity_resource /path/to/your/data true/false true/false tos_config_path")
        sys.exit(1)

    root = sys.argv[1]
    hdfs_loading_str = sys.argv[2]
    tos_load_str = sys.argv[3]
    tos_config_path = sys.argv[4]
    extra_dir = os.path.join(root, "extra")
    hdfs_loading = True if hdfs_loading_str.lower() == "true" else False
    tos_load = True if tos_load_str.lower() == "true" else False


    for split in (EntityResource.Split.TRAIN, EntityResource.Split.VAL):
        print(f"Dumping extra files for {split.value} split...")
        ds = EntityResource(
            root=root, split=split, 
            extra=extra_dir, 
            hdfs_load=hdfs_loading, 
            tos_load=tos_load, 
            tos_config_path=tos_config_path
        )
        ds.dump_extra()
    print(f"✅ All extra files are ready in {extra_dir}")
