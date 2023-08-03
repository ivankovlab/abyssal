import numpy as np
import pandas as pd
import esm
import logging
import torch
import gc

logger = logging.getLogger(__name__)

def check_format(mutation, sequence):
    aa_lst = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    wt_aa = mutation[0]
    position = mutation[1:-1]
    mut_aa = mutation[-1]
    check = True
    # check mutation format
    if not (wt_aa.isalpha() and mut_aa.isalpha() and wt_aa in aa_lst and mut_aa in aa_lst and position.isdigit()):
        logger.warning(f'Wrong mutation format: {mutation}. Mutation should be specified in the format <original amino acid><position><new amino acid> (e.g. M21A)')
        check = False
    else:
        # check that sequence consits of standard amino acids
        if False in [i in aa_lst for i in set(sequence)]:
            logger.warning('Unrecognised amino acids in the sequence. Check the sequence.')
            check = False
        # check that position exists in the sequence
        elif int(position)-1 > len(sequence):
            logger.warning(f'Position {position} does not exist in the sequence.')
            check = False
        elif sequence[int(position)-1] != wt_aa:
            logger.warning("Original amino acid doesn't match the amino acid in the sequence at the specified position :(")
            check = False
    return check

def check_df_format(df):
    if ('mutation' not in df.columns) or ('sequence' not in df.columns):
        logger.warning('The file should contain columns "mutation" and "sequence".')
        return False
    else:
        check = df.apply(lambda x: check_format(x['mutation'], x['sequence']), axis=1)
        if False in check.unique():
            return False
        else:
            return True
        
def load_input(mutation, sequence):
    '''function to load string input similar to Nikita's for loading tsv'''
    position = mutation[1:-1]
    mut_aa = mutation[-1]
    mutant_sequence = sequence[:int(position)-1] + mut_aa + sequence[int(position):]
    return [('_', sequence)], [('_', mutant_sequence)], [int(position)-1], [mutation]

def load_tsv(tsv_file, parse=False):
    '''modified Nikita's function
    the function takes tsv dataset that should obligatory contain columns "mutation" and "sequence"
    the ideas is that user can supply his dataset that may contain other columns without the need to delete them
    the mutated sequence is created by the function'''
    df = pd.read_table(tsv_file)
    if parse == False:
        return df
    else:
        mutate = lambda mutation, sequence: sequence[:int(mutation[1:-1])-1] + mutation[-1] + sequence[int(mutation[1:-1]):]
        df['mutant_sequence'] = df.apply(lambda x: mutate(x['mutation'], x['sequence']), axis=1)
        df['_'] = '_' # а зачем он вообще?
        return  df.loc[:, ['_', 'sequence']].values.tolist(), df.loc[:, ['_', 'mutant_sequence']].values.tolist(), df['mutation'].apply(lambda x: int(x[1:-1]) - 1).tolist(), df['mutation'].tolist()
    
def get_tokens(seqs, alphabet):
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter(seqs)
    return tokens

available_models = {"esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
                    "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,    
                    "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
                    "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D, 	
                    "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D, 	
                    "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D}

def get_embeddings(tokens, positions, esm_model, layer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    esm_model=esm_model.to(device)
    
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(np.split(tokens, len(tokens))):
            if not i%1000 and i!=0:
                logger.info(f"{i+1} embeddings generated")
                torch.cuda.empty_cache()
                gc.collect()
            batch = batch.to(device)
            res = esm_model(batch, repr_layers=[layer])
            resultant_representation = res["representations"][layer][:, positions[i], :]
            embeddings.extend(resultant_representation.cpu())
        logger.info(f"{i+1} embeddings generated")
    return embeddings

def save_embeddings(input_path: str, output_embedding_path: str, esm_model:str, layer: int):
    logger.info('Initializing ESM model')
    esm_model, alphabet = available_models[esm_model]()
    
    logger.info(f'Loading sequences and mutations from {input_path}')
    base_seqs, mutant_seqs, positions, namings = load_tsv(input_path, parse=True)

    logger.info('Initializing base tokens')
    base_tokens = get_tokens(base_seqs, alphabet)
    
    logger.info('Generating base embeddings')
    base_embeddings = get_embeddings(tokens=base_tokens,
                                           positions=positions, 
                                           esm_model=esm_model, 
                                           layer=layer) 
    
    logger.info('Initializing mutant tokens')
    mutant_tokens = get_tokens(mutant_seqs, alphabet)
    
    logger.info('Generating mutant embeddings')
    mutant_embeddings = get_embeddings(tokens=mutant_tokens,
                                           positions=positions, 
                                           esm_model=esm_model, 
                                           layer=layer)

    embeddings = np.array(list(zip(namings, base_embeddings, mutant_embeddings)))
    
    logger.info(f'Saving embeddings to {output_embedding_path}')
    with open(output_embedding_path, 'wb') as outp:
        np.save(outp, embeddings)
    logger.info('Embeddings saved')