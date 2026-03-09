import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from src.data.dataset import CustomDataset

def get_data_loaders_from_split_files(train_csv:str, val_csv:str, batch_size=32, num_workers=4):
    """Load train and validation data from pre-split CSV files (subject-based splits)"""
    train_df = pd.read_csv(train_csv, sep=";")
    val_df = pd.read_csv(val_csv, sep=";")

    transform_data_aug = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(p=0.7),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(train_df, transform=transform_data_aug)
    val_dataset = CustomDataset(val_df, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def get_data_loaders(csv_file:str, split="train", test_size=0.1, batch_size=32, num_workers=4, transform=None, random_state=42):
    df = pd.read_csv(csv_file, sep=";")
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)

    transform_data_aug = v2.Compose([
        # v2.RandomResizedCrop(224, scale=(0.5, 1.0)),  # Random crop + resize
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(p=0.7),               # High flip chance
        # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Strong color changes
        # v2.RandomRotation(30),                        # Random rotation
        # v2.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Shift images
        # v2.RandomGrayscale(p=0.2),                    # Turn to gray sometimes
        # v2.GaussianBlur(kernel_size=(5, 9)),          # Add blur
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_dataset = CustomDataset(train_df, transform=transform_data_aug)
    val_dataset = CustomDataset(val_df, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def get_data_loaders_per_subject(csv_file: str, test_size=0.1, batch_size=32, num_workers=4, random_state=42):

    df = pd.read_csv(csv_file, sep=";")

    # 2. Use GroupShuffleSplit to split by subject (no leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df['subject']))  # ← replace 'subject' with your column name

    # 3. Create train and val dataframes
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(f"train shape: {train_df.shape}")
    print(f"training-subjects: {train_df["subject"].nunique()} \n")

    print(f"val shape: {val_df.shape}")
    print(f"val-subjects: {val_df["subject"].nunique()}")

    # 4. Define image transforms
    transform_train = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(p=0.7),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 5. Create datasets
    train_dataset = CustomDataset(train_df, transform=transform_train)
    val_dataset = CustomDataset(val_df, transform=transform_val)

    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def get_test_data_loader(cfg_data):
    df = pd.read_csv(cfg_data.test_csv, sep=";") # cfg_data
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = CustomDataset(df, transform=transform, return_path=True)
    return DataLoader(
        test_dataset,
        batch_size=cfg_data.batch_size,
        shuffle=False,
        num_workers=cfg_data.num_workers,
        pin_memory=True
    )