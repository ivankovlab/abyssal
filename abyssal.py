import hydra
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
import sys
from os.path import join
from pathlib import Path
import logging

import utils
from nn.dataset import EmbeddingsDataset
from nn.model import AbyssalModel

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
logger = logging.getLogger()

class Abyssal():
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_path = hydra.utils.to_absolute_path(cfg.input)
        self.output_path = hydra.utils.to_absolute_path(cfg.output)
        self.model_path = hydra.utils.to_absolute_path(cfg.get('model_path', 'models/ddg_model_trained_on_mega.pt'))
        
    def check_tsv(self, input_path: str):
        df = utils.load_tsv(input_path, parse=False)
        return utils.check_df_format(df)
    
    def save_embeddings(self, input_tsv_path: str, output_embedding_path: str):
        utils.save_embeddings(input_tsv_path,
                              output_embedding_path,
                              self.cfg.esm_model,
                              self.cfg.layer)
    
    def predict(self, embedding_path: str, model_path: str, output_path: str):
        logger.info('Instantiating model')
        model = AbyssalModel(self.cfg)

        logger.info(f'Loading model from {model_path}')
        state = torch.load(model_path, map_location=None if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        
        logger.info(f'Loading embeddings from {embedding_path}')
        dataset = EmbeddingsDataset(np.load(embedding_path, allow_pickle=True))
        dataloader = DataLoader(dataset, self.cfg.batch_size, shuffle=False)
        
        logger.info('Predicting')
        preds = model.predict(dataloader).squeeze().cpu().numpy()
        
        logger.info(f'Saving output to {output_path}')
        np.savetxt(output_path, preds, fmt='%.3f')
        
        
    def train(self, input_embedding_path: str, output_model_path: str):
        logger.info('Instantiating model')
        accelerator_params = {'devices': 1, 'accelerator': 'gpu'} if torch.cuda.is_available() else {}
        trainer = Trainer(**accelerator_params, enable_checkpointing=False, max_epochs=self.cfg.train.max_epochs)
        model = AbyssalModel(self.cfg)
        
        logger.info('Loading train dataset')
        dataset = EmbeddingsDataset(np.load(input_embedding_path, allow_pickle=True), ground_truth=True)
        train_dataloader = DataLoader(dataset, self.cfg.batch_size, shuffle=True)
        
        logger.info('Training model')
        trainer.fit(model, train_dataloaders=train_dataloader)
        logger.info('Trained!')
        
        
        logger.info(f'Saving model to {output_model_path}')
        torch.save({
                    'model_state_dict': model.state_dict()
                    }, output_model_path)
        logger.info(f'Model saved!')
        
@hydra.main(config_path='configs', config_name='default')
def run_abyssal(cfg):
    abyssal = Abyssal(cfg)
    
    logger.info(f"Running in {cfg.mode} mode")
    # Generate embeddings if input is tsv format
    if abyssal.input_path.split('.')[-1] == 'tsv':
        logger.info('Validating input')
        if abyssal.check_tsv(abyssal.input_path):
            logger.info('Input .tsv file passed sanity checks.')

        else:
            logger.warning('Problems detected in input .tsv file')
        
        Path(hydra.utils.to_absolute_path('embeddings')).mkdir(parents=True, exist_ok=True)
        abyssal.embedding_path = hydra.utils.to_absolute_path(join('embeddings', ''.join(abyssal.input_path.split('/')[-1].split('.')[:-1]) + '.npy'))
        
        if Path(abyssal.embedding_path).is_file():
            logger.info(f"Embeddings file detected at {abyssal.embedding_path}. Using it as input.")
        
        else:
            logger.info(f'Saving embeddings to {abyssal.embedding_path} (this can take a while)')

            abyssal.save_embeddings(abyssal.input_path, abyssal.embedding_path)
            logger.info(f'Embeddings saved to {abyssal.embedding_path}')

    # Otherwise use embeddings directly
    elif cfg.input.split('.')[-1] == 'npy':
        abyssal.embedding_path = abyssal.input_path

    else:
        logger.warning("Input needs to be either a .tsv file with 'sequence' and 'mutation' columns or a .npy file with serialized embeddings. Embeddings are generated automatically from a valid .tsv input and saved.")
    
    if cfg.mode == 'predict':
        abyssal.predict(abyssal.embedding_path, abyssal.model_path, abyssal.output_path)
    elif cfg.mode == 'train':
        abyssal.train(abyssal.embedding_path, abyssal.output_path)
        
if __name__ == '__main__':
    run_abyssal()
