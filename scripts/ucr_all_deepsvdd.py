import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

class GlobalAveragePooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return x.mean(dim=-1)
    

class Upscale(nn.Module):
    def __init__(self, out_channels: int, out_lenght: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.out_lenght = out_lenght
        
    def forward(self, x):
        return x.view(x.size(0), self.out_channels, self.out_lenght)
    
    
from typing import Any, Dict, List, Optional, Union, Tuple

class BaseDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y
        
        assert len(self.x) == len(self.y)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]).float(), torch.from_numpy(np.array([self.y[index]]))


class DeepSVDDAutoEncoder(pl.LightningModule):
    def __init__(self, sequence_length: int, in_channels: int, representation_dim: int = 32) -> None:
        super().__init__()

        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.representation_dim = representation_dim

        # --- Encoder --- #
        self.encoder = nn.Sequential(*[
            nn.Conv1d(in_channels=self.in_channels, out_channels=128, kernel_size=7, bias=False, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, bias=False, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, bias=False, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1),
            GlobalAveragePooling(),
            nn.Linear(in_features=128, out_features=self.representation_dim)
        ])
        
        # --- Decoder --- #
        self.decoder = nn.Sequential(*[
            nn.Linear(in_features=self.representation_dim, out_features=128 * self.sequence_length),
            Upscale(out_lenght=self.sequence_length, out_channels=128),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=3, bias=False, stride=1, dilation=1, padding=3//2),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=3, bias=False, stride=1, dilation=1, padding=3//2),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=5, bias=False, stride=1, dilation=1, padding=5//2),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=7, bias=False, stride=1, dilation=1, padding=7//2),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def configure_optimizers(self) -> Any:
        # Set optimizer for the autoencoder task
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-6, amsgrad=False)
        # Set learning rate scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250], gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        x_hat, z = self(x)
        
        loss = torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim())))
        loss = torch.mean(loss)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> List[Dict[str, Any]]:
        x, y = batch
        x_hat, z = self(x)

        loss = torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim())))
        loss = torch.mean(loss)
        
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return


class DeepSVDD(pl.LightningModule):
    def __init__(self, sequence_length: int, in_channels: int, representation_dim: int = 32) -> None:
        super().__init__()

        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.representation_dim = representation_dim
        
        self.R = torch.tensor(0.0, device=self.device)
        self.nu = 0.1
        self.center = None
        
        self.encoder = nn.Sequential(*[
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=7, bias=False, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, bias=False, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, bias=False, stride=1, dilation=1, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(negative_slope=0.1),
            GlobalAveragePooling(),
            nn.Linear(in_features=128, out_features=32)
        ])

        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z

    def init_center(self, loader: DataLoader, eps: Optional[float] = 0.01) -> torch.Tensor:
        n_samples = 0
        center = torch.zeros(self.representation_dim, device=self.device)

        self.eval()
        with torch.no_grad():
            for (x, y) in loader:
                x = x.to(self.device)
                z = self(x)

                n_samples += z.shape[0]
                center += torch.sum(z, dim=0)

        center /= n_samples

        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center > 0)] = eps
        
        return center
    
    def get_radius(self, distance: torch.Tensor, nu: float):
        return np.quantile(np.sqrt(distance.clone().data.cpu().numpy()), 1 - nu)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-6, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        z = self(x)
        
        distance = torch.sum((z - self.center) ** 2, dim=1)
        scores = distance - self.R ** 2
        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        if self.current_epoch >= 10:
            self.R.data = torch.tensor(self.get_radius(distance, self.nu), device=self.device)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> List[Dict[str, Any]]:
        x, y = batch
        z = self(x)

        distance = torch.sum((z - self.center) ** 2, dim=1)
        scores = distance - self.R ** 2

        preds = torch.max(torch.zeros_like(scores), scores).tolist()
        preds = np.array([1 if pred > 0 else -1 for pred in preds])

        self.log('accuracy_score', accuracy_score(preds, y.cpu().numpy()))
        self.log('f1', f1_score(preds, y.cpu().numpy()))
        self.log('recall', recall_score(preds, y.cpu().numpy()))
        self.log('precision', precision_score(preds, y.cpu().numpy()))

        return


UCR_DATASETS = [
    # 'Adiac',
    # 'ArrowHead',
    # 'Beef',
    # 'BeetleFly',
    # 'BirdChicken',
    # 'Car',
    # 'CBF',
    # 'ChlorineConcentration',
    # 'CinCECGTorso',
    # 'Coffee',
    # 'Computers',
    # 'CricketX',
    # 'CricketY',
    # 'CricketZ',
    # 'DiatomSizeReduction',
    # 'DistalPhalanxOutlineAgeGroup',
    # 'DistalPhalanxOutlineCorrect',
    # 'DistalPhalanxTW',
    # 'Earthquakes',
    # 'ECG200',
    # 'ECG5000',
    # 'ECGFiveDays',
    # 'ElectricDevices',
    # 'FaceAll',
    # 'FaceFour',
    # 'FacesUCR',
    # 'FiftyWords',
    # 'Fish',
    # 'FordA',
    # 'FordB',
    # 'GunPoint',
    # 'Ham',
    # 'HandOutlines',
    # 'Haptics',
    # 'Herring',
    # 'InlineSkate',
    # 'InsectWingbeatSound',
    # 'ItalyPowerDemand',
    # 'LargeKitchenAppliances',
    # 'Lightning2',
    # 'Lightning7',
    # 'Mallat',
    # 'Meat',
    # 'MedicalImages',
    # 'MiddlePhalanxOutlineAgeGroup',
    # 'MiddlePhalanxOutlineCorrect',
    # 'MiddlePhalanxTW',
    # 'MoteStrain',
    # 'NonInvasiveFetalECGThorax1',
    # 'NonInvasiveFetalECGThorax2',
    # 'OliveOil',
    # 'OSULeaf',
    # 'PhalangesOutlinesCorrect',
    # 'Phoneme',
    # 'Plane',
    # 'ProximalPhalanxOutlineAgeGroup',
    # 'ProximalPhalanxOutlineCorrect',
    # 'ProximalPhalanxTW',
    # 'RefrigerationDevices',
    # 'ScreenType',
    # 'ShapeletSim',
    # 'ShapesAll',
    # 'SmallKitchenAppliances',
    # 'SonyAIBORobotSurface1',
    # 'SonyAIBORobotSurface2',
    # 'StarLightCurves',
    # 'Strawberry',
    # 'SwedishLeaf',
    # 'Symbols',
    # 'SyntheticControl',
    # 'ToeSegmentation1',
    # 'ToeSegmentation2',
    # 'Trace',
    # 'TwoLeadECG',
    # 'TwoPatterns',
    # 'UWaveGestureLibraryAll',
    # 'UWaveGestureLibraryX',
    # 'UWaveGestureLibraryY',
    # 'UWaveGestureLibraryZ',
    # 'Wafer',
    # 'Wine',
    # 'WordSynonyms',
    # 'Worms',
    # 'WormsTwoClass',
    # 'Yoga',
    'ACSF1',
    'BME',
    'Chinatown',
    'Crop',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'EthanolLevel',
    'FreezerRegularTrain',
    'FreezerSmallTrain',
    'Fungi',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'HouseTwenty',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'PowerCons',
    'Rock',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'SmoothSubspace',
    'UMD'
]


results = {
    'dataset': [],
    'model': [],
    'label': [],
    'accuracy': [],
    'f1': [],
    'recall': [],
    'precision': [],
}


for dataset in UCR_DATASETS:
    print(f'Starting experiments with {dataset} dataset...')
    # Load the data from .tsv files
    train_data = np.genfromtxt(f'../data/ucr/{dataset}/{dataset}_TRAIN.tsv')
    x_train, y_train = train_data[:, 1:], train_data[:, 0]
    
    test_data = np.genfromtxt(f'../data/ucr/{dataset}/{dataset}_TEST.tsv')
    x_test, y_test = test_data[:, 1:], test_data[:, 0]
    
    unique_labels = np.unique(y_train)
    for label in unique_labels:
        print(f'\tClassifying the label {label}...')
        # Filter samples from positive label
        x_train_ = x_train[y_train == label]
        y_train_ = y_train[y_train == label]

        y_test_ = np.array([1 if y_true == label else -1 for y_true in y_test])
        
        # Apply z normalization
        std_ = x_train_.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train_ = (x_train_ - x_train_.mean(axis=1, keepdims=True)) / std_
        
        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
    
        x_train_ = np.expand_dims(x_train_, axis=1)
        x_test_ = np.expand_dims(x_test, axis=1)

        train_set = BaseDataset(x=x_train_, y=y_train_)
        test_set = BaseDataset(x=x_test_, y=y_test_)
        
        train_loader = DataLoader(train_set, batch_size=16)
        test_loader = DataLoader(test_set, batch_size=16)

        # Train the autoencoder to learn the data representation over the new space
        autoencoder = DeepSVDDAutoEncoder(x_train_.shape[-1], in_channels=1)
        trainer = pl.Trainer(max_epochs=350, accelerator='gpu', devices=-1)
        trainer.fit(autoencoder, train_dataloaders=train_loader)
        
        deepsvdd = DeepSVDD(sequence_length=x_train_.shape[-1], in_channels=1)
        deepsvdd.load_state_dict(autoencoder.state_dict(), strict=False)
        deepsvdd.to(torch.device('cuda'))

        center = deepsvdd.init_center(train_loader)
        deepsvdd.center = center

        trainer_deepsvdd = pl.Trainer(max_epochs=350, accelerator='gpu', devices=-1)
        trainer_deepsvdd.fit(deepsvdd, train_dataloaders=train_loader)
        
        metrics = trainer_deepsvdd.test(deepsvdd, dataloaders=test_loader)[0]
        
        results['dataset'].append(dataset)
        results['model'].append('deepsvdd')
        results['label'].append(label)
        results['accuracy'].append(metrics['accuracy_score'])
        results['f1'].append(metrics['f1'])
        results['recall'].append(metrics['recall'])
        results['precision'].append(metrics['precision'])

metrics = pd.DataFrame(results)
metrics.to_csv('./ucr_deepsvdd.csv', index=False)
