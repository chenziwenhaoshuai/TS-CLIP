import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.interpolate import interp1d
import os
import json


class UCRDataset(Dataset):
    """
    Evaluation dataset for zero-shot time series classification on the UCR Archive.

    Each sample returns a time series tensor, its text embedding, its label ID,
    and the list of class annotation entries (used to map prediction indices back
    to original class IDs).
    """

    def __init__(
        self,
        dataset_name: str,
        ucr_data_path: str = './UCRArchive_2018',
        annotation_path: str = './preprocess_label.json',
        limit_length: int = 200,
    ):
        self.limit_length = limit_length

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # Collect class-level annotations for this dataset
        self.class_annotations = []
        for key in annotation.keys():
            if key.split('_')[0] == dataset_name:
                self.class_annotations.append({
                    'ori_sub_class_id': key.split('_')[1],
                    'prompt_embedding': annotation[key]['prompt_embedding'][0],
                })

        # Load test split
        self.samples = []
        tsv_path = os.path.join(ucr_data_path, dataset_name, f'{dataset_name}_TEST.tsv')
        with open(tsv_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                label = parts[0]
                ts = [float(v) for v in parts[1:]]

                # Replace NaN values with 0
                ts = [0.0 if np.isnan(v) else v for v in ts]

                # Downsample or pad to fixed length
                if len(ts) > self.limit_length:
                    ts = self._interpolate(ts, self.limit_length)
                else:
                    ts = ts + [0.0] * (self.limit_length - len(ts))

                text_embedding = annotation[f'{dataset_name}_{label}']['prompt_embedding'][0]
                total_class_id = annotation[f'{dataset_name}_{label}']['total_class_id']
                self.samples.append({
                    'ts': ts,
                    'text_embedding': text_embedding,
                    'label': label,
                    'total_class_id': total_class_id,
                })

    def _interpolate(self, data, target_length):
        original_indices = np.arange(len(data))
        target_indices = np.linspace(0, len(data) - 1, target_length)
        f = interp1d(original_indices, data, kind='linear', fill_value='extrapolate')
        return f(target_indices).tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ts = torch.tensor(sample['ts'], dtype=torch.float32)
        text_embedding = torch.tensor(sample['text_embedding'], dtype=torch.float32)
        label = torch.tensor(int(sample['label']), dtype=torch.long)
        return ts, text_embedding, label, self.class_annotations

    @staticmethod
    def collate_fn(batch):
        ts_list, text_list, label_list, annos = zip(*batch)
        padded_ts = pad_sequence(ts_list, batch_first=True, padding_value=0.0)
        text_embeddings = torch.stack(text_list, dim=0)
        labels = torch.stack(label_list, dim=0)
        return padded_ts, text_embeddings, labels, annos

