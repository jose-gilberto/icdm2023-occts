from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from typing import Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')


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



class LeakySineLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(0.1 * ((torch.sin(x) ** 2) + x), (torch.sin(x) ** 2) + x)


class ResidualModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convolution_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            padding='same',
            stride=1,
            bias=False
        )
        self.batchnorm_1 = nn.BatchNorm1d(num_features=out_channels)

        self.convolution_2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding='same',
            stride=1,
            bias=False
        )
        self.batchnorm_2 = nn.BatchNorm1d(num_features=out_channels)

        self.convolution_3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding='same',
            stride=1,
            bias=False
        )
        self.batchnorm_3 = nn.BatchNorm1d(num_features=out_channels)
        
        self.activation = LeakySineLU()
        
        self.shortcut = nn.Sequential(*[
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(num_features=out_channels)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.activation(self.batchnorm_1(self.convolution_1(x)))
        x_ = self.activation(self.batchnorm_2(self.convolution_2(x_)))
        x_ = self.batchnorm_3(self.convolution_3(x_))

        return self.activation(x_ + self.shortcut(x))
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_modules: int) -> None:
        super().__init__()

        self.residual_modules = nn.ModuleList()
        self.residual_modules.append(
            ResidualModule(in_channels=in_channels, out_channels=out_channels)
        )
        c_in, c_out = out_channels, out_channels * 2
        for i in range(num_modules - 1):
            self.residual_modules.append(
                ResidualModule(in_channels=c_in, out_channels=c_out)
            )
            c_in = c_out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.residual_modules:
            x = module(x)
        return x


class ResidualEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, representation_dim: int) -> None:
        super().__init__()
        self.residual_net = ResidualBlock(in_channels=in_channels, out_channels=out_channels, num_modules=3)
        self.fc = nn.Linear(in_features=out_channels * 2, out_features=representation_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_net(x) # ResNetEncoder
        x = torch.mean(x, dim=-1) # GAP
        return self.fc(x)


class ResidualTransposeModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convolution_1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=3 // 2,
            stride=1,
            bias=False
        )
        self.batchnorm_1 = nn.BatchNorm1d(num_features=out_channels)

        self.convolution_2 = nn.ConvTranspose1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=5 // 2,
            stride=1,
            bias=False
        )
        self.batchnorm_2 = nn.BatchNorm1d(num_features=out_channels)

        self.convolution_3 = nn.ConvTranspose1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=7,
            padding=7 // 2,
            stride=1,
            bias=False
        )
        self.batchnorm_3 = nn.BatchNorm1d(num_features=out_channels)
        
        self.activation = LeakySineLU()
        
        self.shortcut = nn.Sequential(*[
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(num_features=out_channels)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.activation(self.batchnorm_1(self.convolution_1(x)))
        x_ = self.activation(self.batchnorm_2(self.convolution_2(x_)))
        x_ = self.batchnorm_3(self.convolution_3(x_))

        return self.activation(x_ + self.shortcut(x))


class ResidualTransposeBlock(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: List[int], num_modules: int) -> None:
        super().__init__()
        self.residual_modules = nn.ModuleList()
        for i in range(num_modules):
            self.residual_modules.append(
                ResidualTransposeModule(in_channels=in_channels[i], out_channels=out_channels[i])
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.residual_modules:
            x = module(x)
        return x
    

class ResidualDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, representation_dim: int, sequence_length: int) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.sequence_length = in_channels, out_channels, sequence_length
        self.residual_net = ResidualTransposeBlock(
            in_channels=[in_channels * 2, in_channels * 2, in_channels], out_channels=[in_channels * 2, in_channels, out_channels], num_modules=3)
        
        self.upsample = nn.Linear(in_features=in_channels * 2, out_features=in_channels * 2 * sequence_length, bias=False)
        self.fc = nn.Linear(in_features=representation_dim, out_features=in_channels * 2, bias=False)
        
        self.activation = LeakySineLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x) # representation -> out_channels

        x = self.upsample(x) # out_channels -> out_channels * sequence_len
        x = x.reshape((x.shape[0], self.in_channels * 2, self.sequence_length))
        x = self.activation(x)

        x = self.residual_net(x)
        return x
        

class ResOCAutoEncoder(pl.LightningModule):
    def __init__(self, sequence_length: int, representation_dim: int) -> None:
        super().__init__()
        
        self.encoder = ResidualEncoder(in_channels=1, out_channels=64, representation_dim=representation_dim)
        self.decoder = ResidualDecoder(in_channels=64, out_channels=1, representation_dim=representation_dim, sequence_length=sequence_length)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)
        return [optimizer], [scheduler]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        x_hat = self(x)
        
        loss = torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim())))
        loss = torch.mean(loss)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        x_hat = self(x)

        loss = torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim())))
        loss = torch.mean(loss)

        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return
    

class ResOC(pl.LightningModule):
    def __init__(self, sequence_length: int, representation_dim: int) -> None:
        super().__init__()
        self.representation_dim = representation_dim
        self.encoder = ResidualEncoder(in_channels=1, out_channels=64, representation_dim=representation_dim)
        self.decoder = ResidualDecoder(in_channels=64, out_channels=1, representation_dim=representation_dim, sequence_length=sequence_length)
        
        self.center = None
        self.R = torch.tensor(0)
        self.nu = 0.1
        self.warmup_epochs = 10
        self.sequence_length = sequence_length
        
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        return [optimizer], [scheduler]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z, self.decoder(z)

    def init_center(self, dataloader: DataLoader, eps: float = 0.01):
        n_samples = 0
        c = torch.zeros(self.representation_dim, device=self.device)
        
        self.eval()
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                z, x_hat = self(x)
                n_samples += z.shape[0]
                c += torch.sum(z, dim=0)
                
        c /= n_samples

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c
        
    def get_radius(self, distance: torch.Tensor):
        radius = torch.tensor(np.quantile(np.sqrt(distance.clone().data.cpu().numpy()), 1 - self.nu))
        return radius
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        assert self.center is not None, 'You need to execute init_center method before training the algorithm.'

        x, y = batch
        self.center = self.center.to(self.device)
        z, x_hat = self(x)

        distances = torch.sum((z - self.center) ** 2, dim=1)
        scores = distances - self.R ** 2
        # loss_1 = self.R ** 2 + torch.mean(torch.max(torch.zeros_like(scores), scores))
        # loss_1 = torch.mean(distances)

        loss_2 = torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim())))
        loss_2 = torch.mean(loss_2) # / self.sequence_length
        
        
        loss_1 = self.R ** 2 * torch.mean(torch.max(torch.zeros_like(scores), scores))

        loss = loss_1  + loss_2
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        
        if self.current_epoch >= self.warmup_epochs:
            self.R.data = self.get_radius(distances).to(self.device)

        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        z, x_hat = self(x)

        distances = torch.sum((z - self.center) ** 2, dim=1)
        scores = distances - self.R ** 2

        preds = torch.max(torch.zeros_like(scores), scores).tolist()
        preds = np.array([1 if pred > 0 else -1 for pred in preds])

        self.log('accuracy_score', accuracy_score(preds, y.cpu().numpy()))
        self.log('f1', f1_score(preds, y.cpu().numpy()))
        self.log('recall', recall_score(preds, y.cpu().numpy()))
        self.log('precision', precision_score(preds, y.cpu().numpy()))

        return


UCR_DATASETS = [
    'Adiac',
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
    # 'ACSF1',
    # 'BME',
    # 'Chinatown',
    # 'Crop',
    # 'EOGHorizontalSignal',
    # 'EOGVerticalSignal',
    # 'EthanolLevel',
    # 'FreezerRegularTrain',
    # 'FreezerSmallTrain',
    # 'Fungi',
    # 'GunPointAgeSpan',
    # 'GunPointMaleVersusFemale',
    # 'GunPointOldVersusYoung',
    # 'HouseTwenty',
    # 'InsectEPGRegularTrain',
    # 'InsectEPGSmallTrain',
    # 'MixedShapesRegularTrain',
    # 'MixedShapesSmallTrain',
    # 'PigAirwayPressure',
    # 'PigArtPressure',
    # 'PigCVP',
    # 'PowerCons',
    # 'Rock',
    # 'SemgHandGenderCh2',
    # 'SemgHandMovementCh2',
    # 'SemgHandSubjectCh2',
    # 'SmoothSubspace',
    # 'UMD'
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
        x_test_ = (x_test - x_test.mean(axis=1, keepdims=True)) / std_
    
        x_train_ = np.expand_dims(x_train_, axis=1)
        x_test_ = np.expand_dims(x_test_, axis=1)

        train_set = BaseDataset(x=x_train_, y=y_train_)
        test_set = BaseDataset(x=x_test_, y=y_test_)
        
        train_loader = DataLoader(train_set, batch_size=16)
        test_loader = DataLoader(test_set, batch_size=16)

        # Train the autoencoder to learn the data representation over the new space
        residual_autoencoder = ResOCAutoEncoder(x_train_.shape[-1], representation_dim=32)
        trainer = pl.Trainer(accelerator='gpu', devices=-1, max_epochs=350)
        trainer.fit(residual_autoencoder, train_dataloaders=train_loader)
        
        resoc = ResOC(sequence_length=x_train_.shape[-1], representation_dim=32)
        resoc.load_state_dict(residual_autoencoder.state_dict(), strict=False)
        resoc.init_center(train_loader)

        trainer = pl.Trainer(accelerator='gpu', devices=-1, max_epochs=350)
        trainer.fit(resoc, train_dataloaders=train_loader)

        metrics = trainer.test(resoc, dataloaders=test_loader)[0]
        
#         results['dataset'].append(dataset)
#         results['model'].append('resoc')
#         results['label'].append(label)
#         results['accuracy'].append(metrics['accuracy_score'])
#         results['f1'].append(metrics['f1'])
#         results['recall'].append(metrics['recall'])
#         results['precision'].append(metrics['precision'])

# metrics = pd.DataFrame(results)
# metrics.to_csv('./ucr_resoc.csv', index=False)


