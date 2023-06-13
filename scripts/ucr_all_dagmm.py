from typing import Any, Optional, Tuple
import pandas as pd
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import warnings
warnings.filterwarnings('ignore')
from torch.nn import functional as F
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


class DAGMM(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
    ) -> None:
        super().__init__()

        # Compreension Network
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=60),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=30),
            nn.Tanh(),
            nn.Linear(in_features=30, out_features=10),
            nn.Tanh(),
            nn.Linear(in_features=10, out_features=1),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=10),
            nn.Tanh(),
            nn.Linear(in_features=10, out_features=30),
            nn.Tanh(),
            nn.Linear(in_features=30, out_features=60),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=in_features)
        )
        
        # Estimation network
        self.estimation = nn.Sequential(
            nn.Linear(in_features=3, out_features=10),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=10, out_features=2),
            nn.Softmax(),
        )
        
        self.lambda1 = 0.1
        self.lambda2 = 0.005
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [5, 8], 0.1
        )
        return [optimizer], [scheduler]
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        
        reconstruction_cosine = F.cosine_similarity(x, x_hat, dim=1)
        reconstruction_euclidean = F.pairwise_distance(x, x_hat, p=2)
        
        z = torch.cat([
            z_c,
            reconstruction_euclidean.unsqueeze(-1),
            reconstruction_cosine.unsqueeze(-1)
        ], dim=1)
        
        gamma = self.estimation(z)
        return z_c, x_hat, z, gamma
    
    def reconstruction_error(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        e = torch.tensor(0.0).to(self.device)
        for i in range(x.shape[0]):
            e += torch.dist(x[i], x_hat[i])
        return e / x.shape[0]
    
    def get_gmm_param(self, gamma: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        N = gamma.shape[0]
        ceta = torch.sum(gamma, dim=0) / N
        
        mean = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0)
        mean = mean / torch.sum(gamma, dim=0).unsqueeze(-1)
        
        z_mean = (z.unsqueeze(1) - mean.unsqueeze(0))
        cov = (
            torch.sum(
                gamma.unsqueeze(-1).unsqueeze(-1) * z_mean.unsqueeze(-1) * z_mean.unsqueeze(-2),
                dim=0
            ) /
            torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)
        )
        
        return ceta, mean, cov

    def sample_energy(
        self,
        ceta: torch.Tensor,
        mean: torch.Tensor,
        cov: torch.Tensor,
        zi: torch.Tensor,
        n_gmm: torch.Tensor,
        bs: torch.Tensor
    ) -> torch.Tensor:
        e = torch.tensor(0.0).to(self.device)
        cov_eps = (torch.eye(mean.shape[1]) * (1e-12)).to(self.device)

        for k in range(n_gmm):
            miu_k = mean[k].unsqueeze(1)
            d_k = zi - miu_k
            
            inv_cov = torch.inverse(cov[k] + cov_eps).to(self.device)
            e_k = torch.exp(-0.5 * torch.chain_matmul(torch.t(d_k), inv_cov, d_k))
            e_k = e_k / torch.sqrt(torch.abs(torch.det(2 * math.pi * cov[k])))
            e_k = e_k * ceta[k]
            e += e_k.squeeze()
            
        return -torch.log(e)
    
    def calculate_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        gamma: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, n_gmm = gamma.shape[0], gamma.shape[1]

        # Reconstruction error
        reconstruction_error = self.reconstruction_error(x, x_hat)
        
        # 2nd Loss Part
        ceta, mean, cov = self.get_gmm_param(gamma, z)
        
        # 3nd Loss Part
        e = torch.tensor(0.0).to(self.device)
        for i in range(z.shape[0]):
            z_i = z[i].unsqueeze(1)
            e_i = self.sample_energy(ceta, mean, cov, z_i, n_gmm, bs).to(self.device)
            e += e_i
            
        p = torch.tensor(0.0).to(self.device)
        for k in range(n_gmm):
            cov_k = cov[k]
            p_k = torch.sum(1 / torch.diagonal(cov_k, 0))
            p += p_k
        
        loss = reconstruction_error + (self.lambda1 / z.shape[0]) * e + self.lambda2 * p
        return loss, reconstruction_error, e / z.shape[0], p
    
    def compute_threshold(self, dataloader: DataLoader, len_data: int):
        energies = np.zeros(shape=(len_data))
        step = 0

        self.eval()
        with torch.no_grad():
            for x, y in dataloader:
                z_c, x_hat, z, gamma = self(x)
                m_prob, m_mean, m_cov = self.get_gmm_param(gamma, z)
                
                for i in range(z.shape[0]):
                    z_i = z[i].unsqueeze(1)
                    sample_energy = self.sample_energy(
                        m_prob, m_mean, m_cov, z_i, gamma.shape[1], gamma.shape[0]
                    )
                    
                    energies[step] = sample_energy.detach().item()
                    step += 1

        threshold = np.percentile(energies, 80)
        return threshold
    
    def set_threshold(self, dataloader: DataLoader, len_data: int) -> None:
        self.threshold = self.compute_threshold(dataloader, len_data)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, labels = batch
        z_c, x_hat, z, gamma = self(x)

        loss, reconstruction_error, e, p = self.calculate_loss(
            x=x, x_hat=x_hat, gamma=gamma, z=z
        )

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_recon_error', reconstruction_error, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_energy', e, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        assert self.threshold != None, 'You must set an threshold before perform test steps.'

        x, y = batch
        scores = np.zeros(shape=(len(x), 2))
        step = 0
        
        z_c, x_hat, z, gamma = self(x)
        m_prob, m_mean, m_cov = self.get_gmm_param(gamma, z)

        # For each instance in the batch
        for i in range(z.shape[0]):
            z_i = z[i].unsqueeze(1)
            sample_energy = self.sample_energy(m_prob, m_mean, m_cov, z_i, gamma.shape[1], gamma.shape[0])
            se = sample_energy.detach().item()
            
            scores[step] = [int(y[i]), int(se > self.threshold)]
            step += 1

        accuracy = accuracy_score(scores[:, 0], scores[:, 1])
        precision, recall, f1, support = precision_recall_fscore_support(scores[:, 0], scores[:, 1], average='binary')

        self.log('accuracy', accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log('f1', f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('recall', recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log('precision', precision, on_epoch=True, on_step=False, prog_bar=True)

        return


UCR_DATASETS = [
    'Adiac',
    'ArrowHead',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'Car',
    'CBF',
    'ChlorineConcentration',
    'CinCECGTorso',
    'Coffee',
    'Computers',
    'CricketX',
    'CricketY',
    'CricketZ',
    'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW',
    'Earthquakes',
    'ECG200',
    'ECG5000',
    'ECGFiveDays',
    'ElectricDevices',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    'FiftyWords',
    'Fish',
    'FordA',
    'FordB',
    'GunPoint',
    'Ham',
    'HandOutlines',
    'Haptics',
    'Herring',
    'InlineSkate',
    'InsectWingbeatSound',
    'ItalyPowerDemand',
    'LargeKitchenAppliances',
    'Lightning2',
    'Lightning7',
    'Mallat',
    'Meat',
    'MedicalImages',
    'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW',
    'MoteStrain',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    'OliveOil',
    'OSULeaf',
    'PhalangesOutlinesCorrect',
    'Phoneme',
    'Plane',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect',
    'ProximalPhalanxTW',
    'RefrigerationDevices',
    'ScreenType',
    'ShapeletSim',
    'ShapesAll',
    'SmallKitchenAppliances',
    'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2',
    'StarLightCurves',
    'Strawberry',
    'SwedishLeaf',
    'Symbols',
    'SyntheticControl',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    'TwoLeadECG',
    'TwoPatterns',
    'UWaveGestureLibraryAll',
    'UWaveGestureLibraryX',
    'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ',
    'Wafer',
    'Wine',
    'WordSynonyms',
    'Worms',
    'WormsTwoClass',
    'Yoga',
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

        # In the case of dagmm the normal class is listed as 0 and abnormal as 1
        y_test_ = np.array([0 if y_true == label else 1 for y_true in y_test])

        # Apply z normalization
        std_ = x_train_.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train_ = (x_train_ - x_train_.mean(axis=1, keepdims=True)) / std_
        
        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test_ = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        train_set = BaseDataset(x=x_train_, y=y_train_)
        test_set = BaseDataset(x=x_test_, y=y_test_)
        
        train_loader = DataLoader(train_set, batch_size=16)
        test_loader = DataLoader(test_set, batch_size=16)

        dagmm = DAGMM(x_train_.shape[-1])
        trainer = pl.Trainer(max_epochs=300, accelerator='gpu', devices=-1)
        trainer.fit(dagmm, train_dataloaders=train_loader)
        
        dagmm.set_threshold(train_loader, len(train_set))

        metrics = trainer.test(dagmm, dataloaders=test_loader)[0]

        results['dataset'].append(dataset)
        results['model'].append('dagmm')
        results['label'].append(label)
        results['accuracy'].append(metrics['accuracy'])
        results['f1'].append(metrics['f1'])
        results['recall'].append(metrics['recall'])
        results['precision'].append(metrics['precision'])

metrics = pd.DataFrame(results)
metrics.to_csv('./ucr_dagmm.csv', index=False)
