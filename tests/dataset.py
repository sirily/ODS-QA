import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

import re
import compress_fasttext

USER_PATTERN = re.compile('<@U[A-Za-z0-9]+>')
EMOJI_PATTERN = re.compile(':[A-Za-z0-9_]+:')
LINK_PATTERN = re.compile('<https?://[^\"\s>]+>')
NON_WORD_PATTERN = re.compile('\W')

def pad_tensor(vec, length, dim, pad_symbol):
    """
    Pads a vector ``vec`` up to length ``length`` along axis ``dim`` with pad symbol ``pad_symbol``.
    """
    pad_size = list(vec.shape)
    pad_size[dim] = length - vec.size(dim)
    return torch.cat([vec, torch.tensor(np.full(pad_size, pad_symbol), dtype=torch.float)], dim=dim)

class QAPadder:
    def __init__(self, dim=0, pad_symbol='<PAD>'):
        self.dim = dim
        self.pad_symbol = pad_symbol
        
    def __call__(self, batch):
        # find longest sequence
        max_len_q = max(map(lambda x: x[0][0].shape[self.dim], batch))
        max_len_a = max(map(lambda x: x[0][1].shape[self.dim], batch))
        max_len = max(max_len_q, max_len_a)
        # pad according to max_len
        q_batch = []
        a_batch = []
        label = []
        for (x, y), lab in batch:
            q_batch.append(pad_tensor(vec=x, length=max_len, dim=self.dim, pad_symbol=self.pad_symbol))
            a_batch.append(pad_tensor(vec=y, length=max_len, dim=self.dim, pad_symbol=self.pad_symbol))
            label.append(lab)
        # stack all
        q_batch = torch.stack(q_batch, dim=0)
        a_batch = torch.stack(a_batch, dim=0)
        label = torch.stack(label, dim=0)
        return q_batch, a_batch, label
    
class Dataset(TorchDataset):
    
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, w2v):
        self.data = data
        self.w2v = w2v

    def __getitem__(self, index):
        """
        Returns one tensor pair (question and answer). 
        
        Fasttext model https://github.com/avidale/compress-fasttext/releases/tag/v0.0.1 used for embeddings.        
        """            
        question = torch.tensor([self.w2v[word] for word in self.tokenize(self.normalize(self.data.iloc[index, 0]))], dtype=torch.float)
        if len(question.size()) == 1:
            question = torch.tensor([self.w2v['пусто']], dtype=torch.float)
        answer = torch.tensor([self.w2v[word] for word in self.tokenize(self.normalize(self.data.iloc[index, 1]))], dtype=torch.float)
        if len(answer.size()) == 1:
            answer = torch.tensor([self.w2v['пусто']], dtype=torch.float)
        label = torch.tensor(self.data.iloc[index, 2])

        return (question, answer), label

    def __len__(self):
        """
        == YOUR CODE HERE ==
        """
        return len(self.data)
    
    def normalize(self, text):
        text = re.sub(LINK_PATTERN, 'ссылка', text)
        text = re.sub(USER_PATTERN, ' ', text)
        text = re.sub(EMOJI_PATTERN, 'эмоция', text)
        text = re.sub(NON_WORD_PATTERN, ' ', text)
        text = text.lower()
        return text
    
    def tokenize(self, text):
        return [token for token in text.split()]
    
    
def get_dataloader(data, batch_size):
    """Returns PyTorch Dataloader for ODS_QA binary classification dataset
    """
    ft = compress_fasttext.models.CompressedFastTextKeyedVectors.load("models/ft_freqprune_100K_20K_pq_300.bin")
    ds = Dataset(data, ft)
    
    return DataLoader(ds, collate_fn=QAPadder(dim=0, pad_symbol=0.), batch_size=batch_size, shuffle=True)