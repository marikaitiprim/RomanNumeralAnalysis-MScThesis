from chordgnn.utils import load_score_hgraph, hetero_graph_from_note_array, select_features, HeteroScoreGraph
from chordgnn.contrastive_learning.chord_representations import time_divided_tsv_to_part
import torch
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import random
from chordgnn.data.dataset import BuiltinDataset, chordgnnDataset


class AugmentedNetChordDataset:
    r"""The AugmentedNet Chord Dataset.

    This class collects all the `.tsv` chord annotation files
    from a local folder (recursively) and stores them in `self.scores`.

    Parameters
    ----------
    raw_dir : str
        Local path to the dataset root directory.
    subset : str
        If specified, only files in folders ending with this subset (e.g. "train") are used.
    """

    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.scores = []
        self.process()

    def process(self):
        """Collect all .tsv files in the dataset."""
        root = self.raw_dir
        for file in os.listdir(self.raw_dir):
            if file.endswith(".tsv"):
                full_path = os.path.join(root, file)
                self.scores.append(full_path)

    def get_scores(self):
        """Return list of all valid .tsv file paths."""
        return self.scores


class ChordGraphDataset(chordgnnDataset):
    def __init__(self, dataset_base, max_size=None, verbose=True, nprocs=1, name=None, raw_dir=None):
        self.dataset_base = dataset_base
        self.dataset_base.process()
        self.max_size = max_size
        if verbose:
            print("Loaded AugmentedNetChordDataset Successfully, now processing...")
        self.graph_dicts = list()
        self.n_jobs = nprocs
        super(ChordGraphDataset, self).__init__(
            name=name,
            raw_dir=raw_dir,
            verbose=verbose)

    def process(self):
        Parallel(self.n_jobs)(delayed(self._process_score)(fn) for fn in
                              tqdm(self.dataset_base.scores, desc="Processing AugmentedNetChordGraphDataset"))
        self.load()

    def _process_score(self, score_fn):
        pass
    def has_cache(self):
        # return True
        if all([
            os.path.exists(os.path.join(self.save_path, os.path.splitext(os.path.basename(path))[0])) for path
            in
            self.dataset_base.scores]):
            return True
        return False

    def save(self):
        """save the graph list and the labels"""
        pass

    def load(self):
        self.graphs = list()
        for fn in os.listdir(self.save_path):
            path = os.path.join(self.save_path, fn)
            graph = load_score_hgraph(path, fn)
            self.graphs.append(graph)

    @property
    def features(self):
        return self.graphs[0].x.shape[-1]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return [
            self.get_graph_attr(i)
            for i in idx
        ]

    def get_graph_attr(self, idx):
        if self.graphs[idx].x.size(0) > self.max_size:
            random_idx = random.randint(0, self.graphs[idx].x.size(0) - self.max_size)
            indices = torch.arange(random_idx, random_idx + self.max_size)
            edge_indices = torch.isin(self.graphs[idx].edge_index[0], indices) & torch.isin(
                self.graphs[idx].edge_index[1], indices)
            onset_divs = torch.tensor(
                self.graphs[idx].note_array["onset_div"][random_idx:random_idx + self.max_size])
            unique_onsets = torch.unique(torch.tensor(self.graphs[idx].note_array["onset_div"]), sorted=True)
            label_idx = (unique_onsets >= onset_divs.min()) & (unique_onsets <= onset_divs.max())
            return [
                self.graphs[idx].x[indices],
                self.graphs[idx].edge_index[:, edge_indices] - random_idx,
                self.graphs[idx].edge_type[edge_indices],
                self.graphs[idx].y[label_idx],
                onset_divs,
                self.graphs[idx].name
            ]

        else:
            return [
                self.graphs[idx].x,
                self.graphs[idx].edge_index,
                self.graphs[idx].edge_type,
                self.graphs[idx].y,
                torch.tensor(self.graphs[idx].note_array["onset_div"]),
                self.graphs[idx].name
            ]


def data_to_graph(note_array, name, save_path):
    nodes, edges = hetero_graph_from_note_array(note_array=note_array)
    note_features = select_features(note_array, "chord")
    
    hg = HeteroScoreGraph(
        note_features,
        edges,
        name=name,
        note_array=note_array,
    )
    
    hg.save(save_path)
    del hg, note_array, nodes, edges, note_features
    return


class AugmentedNetChordGraphDataset(ChordGraphDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, num_tasks=11, max_size=512):
        dataset_base = AugmentedNetChordDataset(raw_dir=raw_dir)

        if isinstance(num_tasks, int):
            if num_tasks <= 6:
                self.tasks = {
                    "localkey": 35, "tonkey": 35, "degree1": 22, "degree2": 22,
                    "quality": 16, "inversion": 4, "root": 35
                }
            elif num_tasks == 11:
                self.tasks = {
                    "localkey": 35, "tonkey": 35, "degree1": 22, "degree2": 22,
                    "quality": 16, "inversion": 4, "root": 35,
                    "romanNumeral": 76, "hrhythm": 2, "pcset": 94, "bass": 35,
                }

        super(AugmentedNetChordGraphDataset, self).__init__(
            dataset_base=dataset_base,
            max_size=max_size,
            nprocs=nprocs,
            name="AugmentedNetChordGraphDataset",
            raw_dir=raw_dir,
            verbose=verbose
        )

    def _process_score(self, score_fn):
        name = os.path.splitext(os.path.basename(score_fn))[0]
        x = time_divided_tsv_to_part(score_fn) 

        for i, note_array in enumerate(x):
            data_to_graph(note_array, (name + "-{}".format(i) if i > 0 else name), save_path=self.save_path)

        return