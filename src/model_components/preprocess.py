import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import numpy as np

# Dataset class
class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y, transform_=None):
        self.transform_ = transform_
        self.X = X.values if hasattr(X, 'values') else X
        self.y = y.values if hasattr(y, 'values') else y
        if self.transform_ is not None:
            self.X = self.transform_.transform(self.X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader function
def get_loaders(batch_size=32, test_size=0.2, random_state=42):
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.values if hasattr(heart_disease.data.features, 'values') else heart_disease.data.features
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    y = heart_disease.data.targets.values if hasattr(heart_disease.data.targets, 'values') else heart_disease.data.targets
    y = (y > 0).astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler().fit(X_train)
    train_dataset = HeartDiseaseDataset(X_train, y_train, transform_=scaler)
    print(train_dataset.X)
    print(train_dataset.y)
    test_dataset = HeartDiseaseDataset(X_test, y_test, transform_=scaler)
    print(test_dataset.X)
    print(test_dataset.y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(train_loader)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(test_loader)
    return train_loader, test_loader




get_loaders()



