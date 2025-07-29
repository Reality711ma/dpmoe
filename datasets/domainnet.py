import os
import json
import random
from collections import defaultdict

import pickle
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing


@DATASET_REGISTRY.register()
class DomainNet(DatasetBase):

    dataset_dir = "domainnet"
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_DomainNet.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # Check if split file exists
        if os.path.exists(self.split_path):
            print(f"Reading split from {self.split_path}")
            train, val, test = self.read_split(self.split_path, self.dataset_dir)
        else:
            print(f"Creating new split and saving to {self.split_path}")
            train, val, test = self.read_and_split_data(self.dataset_dir)
            self.save_split(train, val, test, self.split_path, self.dataset_dir)

        # Few-shot settings
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                print(f"Loading few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Class subsampling
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        if subsample != "all":
            train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_split(filepath, image_dir):
        """Read split file and return Datum objects with domain info."""
        with open(filepath, "r") as f:
            split = json.load(f)

        # Build class name to label mapping
        class_names = []
        for paths in split["train"] + split["val"] + split["test"]:
            parts = paths.split("/")
            class_name = parts[1]
            if class_name not in class_names:
                class_names.append(class_name)
        class_names.sort()
        class_name_to_label = {name: i for i, name in enumerate(class_names)}

        # Convert paths to Datum objects
        def _convert(paths):
            items = []
            for p in paths:
                parts = p.split("/")
                domain = parts[0]
                class_name = parts[1]
                impath = os.path.join(image_dir, p)
                label = class_name_to_label[class_name]
                domain_idx = DomainNet.domains.index(domain)
                items.append(Datum(impath=impath, label=label, domain=domain_idx))
            return items

        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        return train, val, test

    @staticmethod
    def read_and_split_data(image_dir):
        """Create train/val/test split from raw data."""
        # Collect all images grouped by class
        class_images = defaultdict(list)
        for domain in DomainNet.domains:
            domain_path = os.path.join(image_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            for class_name in os.listdir(domain_path):
                class_dir = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                # Collect all images with relative paths
                images = [
                    os.path.join(domain, class_name, img)
                    for img in os.listdir(class_dir)
                    if img.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                class_images[class_name].extend(images)

        # Split each class's images
        train = []
        val = []
        test = []
        for class_name, images in class_images.items():
            random.shuffle(images)
            n_total = len(images)
            n_train = int(n_total * 0.5)
            n_val = int(n_total * 0.2)
            n_test = n_total - n_train - n_val

            train.extend(images[:n_train])
            val.extend(images[n_train: n_train + n_val])
            test.extend(images[n_train + n_val:])

        # Create Datum objects
        class_names = sorted(class_images.keys())
        class_name_to_label = {name: i for i, name in enumerate(class_names)}

        def _create_datum(items):
            data = []
            for rel_path in items:
                parts = rel_path.split("/")
                domain = parts[0]
                class_name = parts[1]
                impath = os.path.join(image_dir, rel_path)
                label = class_name_to_label[class_name]
                domain_idx = DomainNet.domains.index(domain)
                data.append(Datum(impath=impath, label=label, domain=domain_idx))
            return data

        return _create_datum(train), _create_datum(val), _create_datum(test)

    @staticmethod
    def save_split(train, val, test, filepath, image_dir):
        """Save split with relative paths."""

        def _extract_relpath(datum):
            return os.path.relpath(datum.impath, image_dir).replace(os.path.sep, "/")

        split = {
            "train": [_extract_relpath(d) for d in train],
            "val": [_extract_relpath(d) for d in val],
            "test": [_extract_relpath(d) for d in test],
        }

        with open(filepath, "w") as f:
            json.dump(split, f, indent=2)

    @staticmethod
    def subsample_classes(train, val, test, subsample="all"):
        """Subsample classes from the dataset."""
        if subsample == "all":
            return train, val, test

        # Get all class names
        all_class_names = set()
        for datum in train + val + test:
            all_class_names.add(DomainNet.get_classname(datum.impath))
        all_class_names = sorted(all_class_names)

        # Select subset
        n_classes = len(all_class_names)
        if subsample == "base":
            n_sub = n_classes // 2
        else:
            n_sub = int(subsample)

        selected = random.sample(all_class_names, n_sub)

        # Filter data
        def _filter(data):
            return [d for d in data if DomainNet.get_classname(d.impath) in selected]

        return _filter(train), _filter(val), _filter(test)

    @staticmethod
    def get_classname(impath):
        """Extract class name from path: .../domain/class_name/image.jpg"""
        parts = impath.split("/")
        return parts[-2]