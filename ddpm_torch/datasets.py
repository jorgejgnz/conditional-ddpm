import os
import csv
import PIL
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, Sampler
from torch.utils.data.distributed import DistributedSampler
from collections import namedtuple

CSV = namedtuple("CSV", ["header", "index", "data"])

def crop_celeba(img):
    return transforms.functional.crop(img, top=40, left=15, height=148, width=148)


class CelebA(datasets.VisionDataset):
    """
    Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
    """
    base_folder = "celeba"

    def __init__(
            self,
            root,
            split,
            download=False,
            transform=transforms.ToTensor(),
            emb_tensor_filename=None
    ):
        super().__init__(root, transform=transform)
        self.split = split
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split.lower()]
        splits = self._load_csv("list_eval_partition.txt", header=0)
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.download = download

        ###########################################
        ## Load emb_tensor
        self.embs_tensor = torch.load(emb_tensor_filename)
        self.c_in_dim = self.embs_tensor.shape[1]

        self.labels = [(4,'Bald'),  #0
            (15,'Eyeglasses'),      #1
            (20,'Male')]            #2

        self.imgs_path = os.path.join(self.root, self.base_folder, "img_align_celeba")
        self.listdir = sorted(os.listdir(self.imgs_path)) # sorted
        ###########################################

    def _load_csv(
            self,
            filename,
            header=None,
    ):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.imgs_path, self.filename[index]))

        if self.transform is not None:
            X = self.transform(X)

        ###########################################
        file_idx = self.listdir.index(self.filename[index])

        emb = self.embs_tensor[file_idx]
        indices = torch.tensor([l[0]for l in self.labels])
        emb = torch.index_select(emb, 0, indices)
        emb = emb.type(torch.float).to(X.device)

        emb = emb / 2.0 + 0.5 # [-1,1] -> [-0.5,0.5] -> [0,1]
        ###########################################

        return X, emb

    def __len__(self):
        return len(self.filename)

    def extra_repr(self):
        lines = ["Split: {split}", ]
        return "\n".join(lines).format(**self.__dict__)


class CustomMNIST(datasets.ImageFolder):

    def __init__(
            self,
            root,
            split,
            transform=transforms.ToTensor(),
    ):
        self.root = os.path.join(root, "CustomMNIST", "images", split) # datasets/CustomMNIST/images/{train,test}
        super().__init__(self.root, transform=transform)
        self.dataset = datasets.ImageFolder(self.root, transform)

    def __getitem__(self, index):        
        x, folder_name = super(datasets.ImageFolder, self).__getitem__(index)
        return x, float(folder_name)

DATA_INFO = {
    "mnist": {
        "data": datasets.MNIST,
        "resolution": (32, 32),
        "channels": 1,
        "transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]),
        "train_size": 60000,
        "test_size": 10000
    },
    "cifar10": {
        "data": datasets.CIFAR10,
        "resolution": (32, 32),
        "channels": 3,
        "transform": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "_transform": transforms.PILToTensor(),
        "train_size": 50000,
        "test_size": 10000
    },
    "celeba": {
        "data": CelebA,
        "resolution": (64, 64),
        "channels": 3,
        "transform": transforms.Compose([
            #crop_celeba,                                                   # Cropped and rescaled in preprocessing to shorten memory read times
            #transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "_transform": transforms.Compose([
            #crop_celeba,
            #transforms.Resize((64, 64)),
            transforms.PILToTensor()
        ]),
        "train": 162770,
        "test": 19962,
        "validation": 19867
    },
    "custom-mnist": {
        "data": CustomMNIST,
        "resolution": (32, 32),
        "channels": 3,
        "transform": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]),
        "train_size": 60000,
        "test_size": 10000
    },
}

ROOT = os.path.expanduser("~/datasets")


def train_val_split(dataset, val_size, random_seed=None):
    train_size = DATA_INFO[dataset]["train_size"]
    if random_seed is not None:
        np.random.seed(random_seed)
    train_inds = np.arange(train_size)
    np.random.shuffle(train_inds)
    val_size = int(train_size * val_size)
    val_inds, train_inds = train_inds[:val_size], train_inds[val_size:]
    return train_inds, val_inds


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_dataloader(
        dataset,
        batch_size,
        split,
        val_size=0.1,
        random_seed=None,
        root=ROOT,
        pin_memory=False,
        drop_last=False,
        num_workers=0,
        distributed=False,

        emb_tensor_filename=None,
        names_np_filename=None
):
    assert isinstance(val_size, float) and 0 <= val_size < 1
    transform = DATA_INFO[dataset]["transform"]
    dataloader_configs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "num_workers": num_workers
    }
    if dataset == "celeba":
        data = DATA_INFO[dataset]["data"](root=root, split=split, transform=transform, emb_tensor_filename=emb_tensor_filename)
    elif dataset == "custom-mnist":
        data = DATA_INFO[dataset]["data"](root=root, split=split, transform=transform)
    else:
        if split == "test":
            data = DATA_INFO[dataset]["data"](
                root=root, train=False, download=False, transform=transform)
        else:
            data = DATA_INFO[dataset]["data"](
                root=root, train=True, download=True, transform=transform)
            if val_size == 0:
                assert split == "train"
            else:
                train_inds, val_inds = train_val_split(dataset, val_size, random_seed)
                data = Subset(data, {"train": train_inds, "valid": val_inds}[split])

    dataloader_configs["sampler"] = sampler = DistributedSampler(
        data, shuffle=True, seed=random_seed, drop_last=drop_last) if distributed else None
    dataloader_configs["shuffle"] = (sampler is None) if split in {"train", "all"} else False
    dataloader = DataLoader(data, **dataloader_configs)
    return dataloader, sampler
