import torch as T
from torch.utils.data import Dataset
import numpy as np


class EmbeddingsDataset(Dataset):
    def __init__(self, source_data, ground_truth=False):
        """
            source_data format : [(ddg, emb_b, emb_m), ...]
            
            ddg - значение ddg
            emb_b - эмбеддинг для оригинальной аминокислоты в i-ой позиции где проводилась мутация
            emb_m - эмбеддинг для мутированной аминокислоты в i-ой позиции где проводилась мутация 
        """
        self.ground_truth = ground_truth
        
        self.embedding_base = T.stack(list(np.take(source_data, 1, axis=1)), axis=0)
        self.embedding_mutation = T.stack(list(np.take(source_data, 2, axis=1)), axis=0)
                
        if self.ground_truth:
            self.ddg = T.tensor(np.take(source_data, 0, axis=1).astype(dtype='f'))
        
    def __len__(self):
        return self.embedding_base.shape[0]  

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        embedding_base = self.embedding_base[idx, 0:]
        embedding_mutation = self.embedding_mutation[idx, 0:]
        sample = { 'embedding_base' : embedding_base,
                    'embedding_mutation' : embedding_mutation}
    
        if self.ground_truth:
            sample['ddg'] = self.ddg[idx]
            
        return sample