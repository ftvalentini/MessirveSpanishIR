
import math
import random
import gzip
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.dataset as pds
import pyarrow.compute as pc
import torch
from datasets import Dataset
from torch import Tensor
from torch.utils.data import Sampler
from transformers import AutoModel


class TextEncoder:
    def __init__(
        self, model_name, device="cuda:0", pooling="mean", fp16=True
    ):
        self.model_name = model_name
        self.device = device
        self.pooling = pooling
        self.fp16 = fp16

    def load_model(self, device_ids=[0]):
        if self.fp16:
            self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float16)
        else:
            self.model = AutoModel.from_pretrained(self.model_name)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pooling(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Source: https://huggingface.co/intfloat/multilingual-e5-large
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, model_inputs: dict) -> Tensor:
        """NOTE this assumes that the inputs have already been tokenized and 
        are in the device.
        """
        outputs = self.model(**model_inputs)
        if self.pooling == "mean":
            embeddings = self._mean_pooling(outputs[0], model_inputs["attention_mask"])
        elif self.pooling == "cls":
            embeddings = outputs[0][:, 0, :]
        else:
            raise ValueError(f"Pooling method {self.pooling} not supported")
        return embeddings


class BatchShufflingSampler(Sampler):
    """Shuffle batches before iterating over them (not the data itself).
    """
    def __init__(self, data_source, batch_size, seed=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        n = len(self.data_source)
        batches = [range(i, min(i + self.batch_size, n)) for i in range(0, n, self.batch_size)]
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return math.ceil(len(self.data_source) / self.batch_size)


class TxtWriter:
    """Write embeddings to a text file (one embedding per line), and ids to 
    a separate file.
    NOTE we use gzip to compress. It allows to append to the file. We checked
    it's similar in size to np.save. We don't use np.save because it's not
    meant to be appended to.
    """
    def __init__(self, dir_path):
        self.vectors_file_name = 'vectors.txt.gz'
        self.id_file_name = 'ids.txt'
        self.dir_path = dir_path
        self.vectors_file = None
        self.id_file = None

    def __enter__(self):
        if not Path(self.dir_path).exists():
            Path(self.dir_path).mkdir(parents=True, exist_ok=True)
        self.id_file = open(str(Path(self.dir_path) / self.id_file_name), 'a')
        self.vectors_file = gzip.open(str(Path(self.dir_path) / self.vectors_file_name), 'at')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.id_file.close()
        self.vectors_file.close()

    def write(self, ids: list, vectors: np.ndarray) -> None:
        """ids: shape (batch_size,)
        vectors: shape (batch_size, dimension)
        """
        self.id_file.write('\n'.join(ids) + '\n')
        vectors_to_write = [' '.join(map(str, vector)) + '\n' for vector in vectors]
        self.vectors_file.writelines(vectors_to_write)

def filter_ids(dataset: Dataset, ids: list, id_field: str) -> Dataset:
    """Remove rows from dataset that have ids in the list.
    NOTE we use: https://github.com/huggingface/datasets/issues/1796#issuecomment-1900423162
    because filter() and select() are really slow or memory inefficient.
    Converting to pandas and back is also memory inefficient.
    """
    mask = (~pd.Series(dataset[id_field]).isin(ids)).to_numpy()
    dataset = dataset.add_column("mask", mask)
    expr = pc.field("mask") == True
    dataset = (
        dataset.with_format("arrow")
        .filter(
            lambda t: pds.dataset(t).to_table(columns={"mask": expr})[0].to_numpy(),
            batched=True,
        )
        .with_format(None)
    )
    dataset = dataset.remove_columns(["mask"])
    return dataset
