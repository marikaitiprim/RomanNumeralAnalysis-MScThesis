from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset
from chordgnn.contrastive_learning.chord import AugmentedNetChordGraphDataset
from chordgnn.utils import add_reverse_edges_from_edge_index
from chordgnn.data.samplers import BySequenceLengthSampler
import numpy as np
from chordgnn.utils.chord_representations import available_representations
from chordgnn.contrastive_learning.chord import PairedContrastiveDataset


class ContrastiveGraphDatamodule(LightningDataModule):
    def __init__(self, batch_size=1, num_workers=4, num_tasks=11):
        super(ContrastiveGraphDatamodule, self).__init__()
        self.bucket_boundaries = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize_features = True
        dataset = '/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_tsv_jhub'
        augmentation_pitch = '/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_pitch_jhub'
        augmentation_time = '/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_time_jhub'


        self.datasets = [
            AugmentedNetChordGraphDataset(raw_dir=dataset, nprocs=self.num_workers,
            num_tasks=num_tasks),
            AugmentedNetChordGraphDataset(raw_dir=augmentation_pitch, nprocs=self.num_workers,
            num_tasks=num_tasks),
            AugmentedNetChordGraphDataset(raw_dir=augmentation_time, nprocs=self.num_workers,
            num_tasks=num_tasks)
        ]

        self.tasks = self.datasets[0].tasks
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        self.features = self.datasets[0].features

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # create the datasets
        self.dataset_train = self.datasets[0]
        self.dataset_train_pitch = self.datasets[1]
        self.dataset_train_time = self.datasets[2]

        self.paired_dataset = PairedContrastiveDataset(
            original_dataset=self.dataset_train,
            pitch_aug_dataset=self.dataset_train_pitch,
            time_aug_dataset=self.dataset_train_time
        )

        print(f"Original dataset: {len(self.dataset_train)}")
        print(f"Pitch augmented dataset: {len(self.dataset_train_pitch)}")
        print(f"Tempo augmented dataset: {len(self.dataset_train_time)}")

    # def collate_fn(self, batch):
    #     batch_inputs, edges, edge_type, onset_div, name = batch[0]
    #     # batch_inputs = F.normalize(batch_inputs.squeeze(0)) if self.normalize_features else batch_inputs.squeeze(0)
    #     batch_inputs = batch_inputs.squeeze(0).float()
    #     onset_div = onset_div.squeeze().to(batch_inputs.device)
      
    #     edges = edges.squeeze(0)
    #     # Add reverse edges
    #     edge_type = edge_type.squeeze(0)
    #     edges, edge_type = add_reverse_edges_from_edge_index(edges, edge_type)
    #     # edges = torch.cat([edges, edges.flip(0)], dim=1)
    #     # edge_type = torch.cat([edge_type, edge_type], dim=0)
    #     return batch_inputs, edges, edge_type, onset_div, name
    
    def collate_fn(self, batch):
        batch_view1 = [example[0] for example in batch]
        batch_view2 = [example[1] for example in batch]
        return self.collate_train_fn(batch_view1), self.collate_train_fn(batch_view2)

    def collate_train_fn(self, examples):
        lengths = list()
        x = list()
        edge_index = list()
        edge_types = list()
        onset_divs = list()
        max_idx = []
        max_onset_div = []
        for e in examples:
            lengths.append(e[3].shape[0])
            x.append(e[0])
            edge_index.append(e[1])
            edge_types.append(e[2])
            onset_divs.append(e[4])
            max_idx.append(e[0].shape[0])
            max_onset_div.append(e[4].max().item() + 1)
        lengths = torch.tensor(lengths).long()
        lengths, perm_idx = lengths.sort(descending=True)
        perm_idx = perm_idx.tolist()
        max_idx = np.cumsum(np.array([0] + [max_idx[i] for i in perm_idx]))
        max_onset_div = np.cumsum(np.array([0] + [max_onset_div[i] for i in perm_idx]))
        x = torch.cat([x[i] for i in perm_idx], dim=0).float()
        edge_index = torch.cat([edge_index[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        edge_types = torch.cat([edge_types[i] for i in perm_idx], dim=0).long()
        onset_divs = torch.cat([onset_divs[pi]+max_onset_div[i] for i, pi in enumerate(perm_idx)], dim=0).long()
        return x, edge_index, edge_types, onset_divs, lengths

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.paired_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True
        )