from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset
from chordgnn.contrastive_learning.chord import AugmentedNetChordGraphDataset
from collections import defaultdict
from chordgnn.utils import add_reverse_edges_from_edge_index
from chordgnn.data.samplers import BySequenceLengthSampler

def idx_tuple_to_dict(idx_tuple, datasets_map):
    """Transforms indices of a list of tuples of indices (dataset, piece_in_dataset) 
    into a dict {dataset: [piece_in_dataset,...,piece_in_dataset]}"""
    result_dict = defaultdict(list)
    for x in idx_tuple:
        result_dict[datasets_map[x][0]].append(datasets_map[x][1])
    return result_dict


class ContrastiveGraphDatamodule(LightningDataModule):
    def __init__(self, batch_size=1, num_workers=4, num_tasks=11):
        super(ContrastiveGraphDatamodule, self).__init__()
        self.bucket_boundaries = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize_features = True
        dataset = '/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_tsv'
        augmentation_pitch = '/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_augmentation_pitch'
        augmentation_time = '/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_augmentation_time'


        self.datasets = [
            AugmentedNetChordGraphDataset(raw_dir=dataset, nprocs=self.num_workers,
            num_tasks=num_tasks),
            AugmentedNetChordGraphDataset( raw_dir=augmentation_pitch, nprocs=self.num_workers,
            num_tasks=num_tasks),
            AugmentedNetChordGraphDataset( raw_dir=augmentation_time, nprocs=self.num_workers,
            num_tasks=num_tasks)
        ]

        self.tasks = self.datasets[0].tasks
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        self.features = self.datasets[0].features

    def prepare_data(self):
        pass

    def setup(self):
        self.datasets_map = [(dataset_i, piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in
                             range(len(dataset))]

        idxs = range(len(self.datasets_map))

        train_idx = [
            i
            for i in idxs
            if self.datasets[self.datasets_map[i][0]].graphs[
                   self.datasets_map[i][1]] == "training"
        ]

        train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)

        # create the datasets
        self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])

        print("Running on all collections with contrastive learning setup")
        print(f"Original dataset: {len(self.datasets[0])} pieces")
        print(f"Pitch augmented dataset: {len(self.datasets[1])} pieces")
        print(f"Tempo augmented dataset: {len(self.datasets[2])} pieces")
        print(f"Train size: {len(self.dataset_train)}, Test size: {len(self.dataset_test)}")

    def collate_fn(self, batch):
        batch_inputs, edges, edge_type, batch_label, onset_div, name = batch[0]
        # batch_inputs = F.normalize(batch_inputs.squeeze(0)) if self.normalize_features else batch_inputs.squeeze(0)
        batch_inputs = batch_inputs.squeeze(0).float()
        batch_labels = batch_label.squeeze(0)
        onset_div = onset_div.squeeze().to(batch_inputs.device)

        from chordgnn.utils.chord_representations import available_representations
      
        batch_label = {task: batch_labels[:, i].squeeze().long() for i, task in enumerate(available_representations.keys())}
        batch_label["onset"] = batch_labels[:, -1].squeeze().long()
        edges = edges.squeeze(0)
        # Add reverse edges
        edge_type = edge_type.squeeze(0)
        edges, edge_type = add_reverse_edges_from_edge_index(edges, edge_type)
        # edges = torch.cat([edges, edges.flip(0)], dim=1)
        # edge_type = torch.cat([edge_type, edge_type], dim=0)
        return batch_inputs, edges, edge_type, batch_label, onset_div, name

    # def collate_train_fn(self, examples):
    #     lengths = list()
    #     x = list()
    #     edge_index = list()
    #     edge_types = list()
    #     y = list()
    #     onset_divs = list()
    #     max_idx = []
    #     max_onset_div = []
    #     for e in examples:
    #         lengths.append(e[3].shape[0])
    #         x.append(e[0])
    #         edge_index.append(e[1])
    #         edge_types.append(e[2])
    #         y.append(e[3])
    #         onset_divs.append(e[4])
    #         max_idx.append(e[0].shape[0])
    #         max_onset_div.append(e[4].max().item() + 1)
    #     lengths = torch.tensor(lengths).long()
    #     lengths, perm_idx = lengths.sort(descending=True)
    #     perm_idx = perm_idx.tolist()
    #     max_idx = np.cumsum(np.array([0] + [max_idx[i] for i in perm_idx]))
    #     max_onset_div = np.cumsum(np.array([0] + [max_onset_div[i] for i in perm_idx]))
    #     x = torch.cat([x[i] for i in perm_idx], dim=0).float()
    #     edge_index = torch.cat([edge_index[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
    #     edge_types = torch.cat([edge_types[i] for i in perm_idx], dim=0).long()
    #     # y = torch.cat([y[i] for i in perm_idx], dim=0).float()
    #     # batch_label = {task: y[:, i].squeeze().long() for i, task in
    #     #                enumerate(available_representations.keys())}
    #     # batch_label["onset"] = y[:, -1]
    #     y = torch.nn.utils.rnn.pad_sequence([y[i] for i in perm_idx], batch_first=True, padding_value=-1)
    #     if self.version == "v1.0.0":
    #         from chordgnn.utils.chord_representations import available_representations
    #     else:
    #         from chordgnn.utils.chord_representations_latest import available_representations
    #     batch_label = {task: y[:, :, i].squeeze().long() for i, task in
    #                    enumerate(available_representations.keys())}
    #     batch_label["onset"] = y[:, :, -1].long()
    #     onset_divs = torch.cat([onset_divs[pi]+max_onset_div[i] for i, pi in enumerate(perm_idx)], dim=0).long()
    #     return x, edge_index, edge_types, batch_label, onset_divs, lengths

    def train_dataloader(self):
        sampler = BySequenceLengthSampler(self.dataset_train, self.bucket_boundaries, self.batch_size)
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_sampler=sampler,
            batch_size=1,
            num_workers=0,
            collate_fn=self.collate_train_fn,
            drop_last=False,
            pin_memory=False,
        )