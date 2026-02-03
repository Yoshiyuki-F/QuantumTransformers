import os
import tarfile
from glob import glob
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import gdown
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset as hf_load_dataset
from transformers import AutoTokenizer

class NumPyFolderDataset(Dataset):
    """
    A dataset consisting of NumPy arrays stored in folders (one folder per class).
    """
    def __init__(self, name, img_shape, num_classes, extracted_data_path=None, gdrive_id=None, data_dir='~/data', transform=None):
        self.name = name
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform

        if extracted_data_path is not None:
            print(f'Using existing data at {extracted_data_path}')
            self.dataset_path = Path(extracted_data_path)
        elif gdrive_id is not None:
            self.dataset_path = Path(self.data_dir) / self.name
            if not self.dataset_path.exists():
                os.makedirs(self.dataset_path, exist_ok=True)
                archive_path = Path(self.data_dir) / f'{self.name}.tar.xz'
                gdown.download(id=gdrive_id, output=str(archive_path), quiet=False)
                with tarfile.open(archive_path, 'r:xz') as f:
                    print(f'Extracting {self.name}.tar.xz to {self.data_dir}')
                    f.extractall(self.data_dir)
                os.remove(archive_path)
        else:
            raise ValueError('Either extracted_data_path or gdrive_id must be provided')

        # Collect all files
        self.files = []
        self.labels = []
        self.class_names = sorted([d.name for d in self.dataset_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        for cls_name in self.class_names:
            cls_dir = self.dataset_path / cls_name
            for f in cls_dir.glob('*.npy'):
                self.files.append(f)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        try:
            image = np.load(file_path).astype(np.float32)
            if self.img_shape is not None:
                 if image.shape != self.img_shape:
                    # Try to transpose if channels are in the wrong place
                    if image.shape[0] == self.img_shape[-1]:
                        image = np.transpose(image, (1, 2, 0))
                    elif image.shape[-1] == self.img_shape[0]:
                        image = np.transpose(image, (2, 0, 1))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a zeroed image as fallback or raise?
            image = np.zeros(self.img_shape, dtype=np.float32)

        if self.transform:
            image = self.transform(image)
        
        # Ensure it's a tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        
        return image, label

def numpy_collate(batch):
    """
    Collate function to convert PyTorch Tensors to NumPy arrays for JAX.
    """
    if isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], torch.Tensor):
        return np.stack([b.numpy() for b in batch])
    elif isinstance(batch[0], (np.ndarray, np.generic)):
        return np.stack(batch)
    else:
        return np.array(batch)

def datasets_to_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, drop_remainder=True, collate_fn=numpy_collate):
    
    loader_kwargs = {
        'batch_size': batch_size,
        'drop_last': drop_remainder,
        'collate_fn': collate_fn,
        'num_workers': 0, # Simplify for now, can increase
        'pin_memory': torch.cuda.is_available()
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

def get_mnist_dataloaders(data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True):
    """
    Returns dataloaders for the MNIST dataset.
    """
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.permute(1, 2, 0) # Convert (C, H, W) to (H, W, C) for JAX/Flax consistency if needed. 
                                     # BUT checking original: dataset_info_from_configs... usually HWC in TF.
                                     # Let's keep strict check. ViT usually wants HWC or CHW?
                                     # efficient_attention in transformers.py expects (batch, seq, hidden) or (batch, H, W, C)?
                                     # SwinT expects (b, h, w, c).
                                     # PyTorch ToTensor produces (C, H, W).
                                     # So we MUST permute to (H, W, C).
    ])
    
    # Load full dataset
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Split train/val (90%/10%)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print("Cardinalities (train, val, test):", len(train_dataset), len(val_dataset), len(test_dataset))

    return datasets_to_dataloaders(train_dataset, val_dataset, test_dataset, batch_size,
                                   drop_remainder=drop_remainder)


def get_electron_photon_dataloaders(data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True):
    data_dir = os.path.expanduser(data_dir)
    
    # Custom splits logic involves downloading first.
    # We can reuse NumPyFolderDataset logic but we need to split manually if it's not structured.
    # The original was downloading via TFDS builder logic.
    # Here we instantiate the dataset which handles download.
    
    full_dataset = NumPyFolderDataset(name="electron-photon", img_shape=(32, 32, 2), num_classes=2, 
                                      gdrive_id="1VAqGQaMS5jSWV8gTXw39Opz-fNMsDZ8e", data_dir=data_dir,
                                      transform=transforms.ToTensor()) 
                                      # Note: np load is (32, 32, 2). ToTensor makes it (2, 32, 32).
                                      # We need (32, 32, 2) for the model?
                                      # SwinT code: `if not self.channels_last: x = x.transpose((0, 3, 1, 2))`
                                      # But default is channels_last=True in VisionTransformer.
                                      # So we generally want HWC.
                                      # ToTensor converts HWC to CHW.
                                      # So we should avoid ToTensor if we want HWC, or permute back.
    
    # Wait, NumPyFolderDataset returns numpy array. transform=ToTensor converts to Tensor (C, H, W).
    # If the file is (32, 32, 2) HWC.
    # Let's define a custom transform to keep HWC but float32.
    
    def to_float_tensor_hwc(img):
        return torch.tensor(img, dtype=torch.float32)

    full_dataset.transform = to_float_tensor_hwc
    
    # Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # For test, usually there is a separate folder or we split?
    # TFDS split was: train[:90%], train[90%:], test.
    # NumPyFolderDataset original split logic used dataset_path / 'train' and 'test'.
    
    # Let's fix NumPyFolderDataset to looking for 'train' and 'test' folders if they exist, or root?
    # Original _split_generators looked for 'train' and 'test' subdirs.
    
    # We need to construct datasets pointing to specific subfolders.
    dataset_root = Path(data_dir) / "electron-photon"
    if not dataset_root.exists():
         # Trigger download by instantiating once (hacky)
         _ = NumPyFolderDataset(name="electron-photon", img_shape=(32, 32, 2), num_classes=2, 
                                      gdrive_id="1VAqGQaMS5jSWV8gTXw39Opz-fNMsDZ8e", data_dir=data_dir)
    
    train_val_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    
    train_val_dataset = NumPyFolderDataset(name="electron-photon", img_shape=(32, 32, 2), num_classes=2,
                                           extracted_data_path=train_val_dir, transform=to_float_tensor_hwc)
    test_dataset = NumPyFolderDataset(name="electron-photon", img_shape=(32, 32, 2), num_classes=2,
                                      extracted_data_path=test_dir, transform=to_float_tensor_hwc)
                                      
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    print("Cardinalities (train, val, test):", len(train_dataset), len(val_dataset), len(test_dataset))
    
    return datasets_to_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, drop_remainder=drop_remainder)

def get_quark_gluon_dataloaders(data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True):
    data_dir = os.path.expanduser(data_dir)
    
    # Trigger download
    dataset_root = Path(data_dir) / "quark-gluon"
    if not dataset_root.exists():
         _ = NumPyFolderDataset(name="quark-gluon", img_shape=(125, 125, 3), num_classes=2, 
                                      gdrive_id="1PL2YEr5V__zUZVuUfGdUvFTkE9ULHayz", data_dir=data_dir)

    def to_float_tensor_hwc(img):
        return torch.tensor(img, dtype=torch.float32)

    train_val_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    
    train_val_dataset = NumPyFolderDataset(name="quark-gluon", img_shape=(125, 125, 3), num_classes=2,
                                           extracted_data_path=train_val_dir, transform=to_float_tensor_hwc)
    test_dataset = NumPyFolderDataset(name="quark-gluon", img_shape=(125, 125, 3), num_classes=2,
                                      extracted_data_path=test_dir, transform=to_float_tensor_hwc)
                                      
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    print("Cardinalities (train, val, test):", len(train_dataset), len(val_dataset), len(test_dataset))
    
    return datasets_to_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, drop_remainder=drop_remainder)

def get_medmnist_dataloaders(dataset: str, data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True):
    raise NotImplementedError

def get_imdb_dataloaders(data_dir: str = '~/data', batch_size: int = 1, drop_remainder: bool = True,
                         max_vocab_size: int = 20_000, max_seq_len: int = 512):
    
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True) # Hugging face cache usually handles this but good to have
    
    # Load dataset using Hugging Face datasets
    dataset = hf_load_dataset("imdb", cache_dir=data_dir)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_len)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Rename 'label' to 'labels' or keep? Model expects 'labels' usually?
    # The existing training code expects (inputs, labels).
    # Tokenizer returns input_ids, token_type_ids, attention_mask.
    # The original TF code returned: (padded_inputs, label)
    # padded_inputs was just IDs.
    
    def format_transforms(examples):
        # Flatten inputs? No, tokenizer output is list of lists
        return {
            "inputs": np.array(examples["input_ids"], dtype=np.int32),
            "label": np.array(examples["label"], dtype=np.int32)
        }
    
    tokenized_datasets.set_format(type="numpy", columns=["input_ids", "label"], output_all_columns=False)

    # Split train to train/val
    train_full = tokenized_datasets["train"]
    # HF dataset split
    train_val_split = train_full.train_test_split(test_size=0.1, seed=42)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]
    test_data = tokenized_datasets["test"]
    
    # Create DataLoaders
    # We need a collate dict?
    # datasets_to_dataloaders collate expects direct inputs/labels list.
    # The HF dataset iteration yields keys.
    
    # Let's wrap HF dataset in a simpler Torch Dataset
    class HFWrapper(Dataset):
        def __init__(self, hf_ds):
            self.hf_ds = hf_ds
        def __len__(self):
            return len(self.hf_ds)
        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            return item["input_ids"], item["label"]
            
    train_dataset = HFWrapper(train_data)
    val_dataset = HFWrapper(val_data)
    test_dataset = HFWrapper(test_data)
    
    print("Cardinalities (train, val, test):", len(train_dataset), len(val_dataset), len(test_dataset))
    
    # Return dataloaders, vocab (dummy size), and tokenizer
    vocab = tokenizer.vocab # This is a dict
    
    return datasets_to_dataloaders(train_dataset, val_dataset, test_dataset, batch_size,
                                   drop_remainder=drop_remainder), vocab, tokenizer

