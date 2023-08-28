# ABYSSAL - &Delta;&Delta;G predictor trained on Mega dataset and ESM2-embeddings

## Usage

Notes on the ABYSSAL repository are given below. For an easy way to use our trained model for generating predictions, see [this colab notebook](https://colab.research.google.com/github/ivankovlab/abyssal/blob/colab/abyssal_predictor_colab.ipynb).

ABYSSAL can be used to predict &Delta;&Delta;G for a user-supplied dataset using a pretrained model, or to train a predictive model on the supplied data. Input data is accepted either as a .tsv file with the original sequences and mutations in `sequence` and `mutation` columns, or a serialized .npy file containing embeddings for the mutation. If a .tsv file is given, the embeddings are computed and saved automatically.

Firstly, clone this repo:

```
git clone https://github.com/ivankovlab/abyssal.git
cd abyssal
```

### Environment

The simplest way to replicate our environment is to use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):

```
conda env create -n abyssal --file environment.yaml
conda activate abyssal
```

### Models

A model pretained on the Mega dataset is available for download at https://zenodo.org/record/7886963 or you can use the command:

```
wget https://zenodo.org/record/7886963/files/ddg_model_trained_on_mega.pt -P ./models
```

### Predicting

To use a pretrained model for &Delta;&Delta;G predictions given a user input:

```
python3 abyssal.py input={input.tsv or input.npy} output={predictions.tsv} model_path={model.pt}
```

Default model path is `models/ddg_model_trained_on_mega.pt`.

Example:

```
python3 abyssal.py input=test/S669_sample.tsv output=test/predictions.tsv
```

The command will create folder `outputs/` with the log of the run. The embeddings will be saved into `embeddings/` folder.

### Training

To train a model on given data and save the generated model:

```
python3 abyssal.py mode=train input={input.tsv or input.npy} output={output_model.pt}
```

### Configuration

The parameters of the model and training process can be modified in `configs/default.yaml`.

## Data

### Mega dataset processing

`notebooks/01_processing.ipynb` - processing and EDA of Mega dataset from `Processed_K50_dG_datasets/K50_dG_Dataset1_Dataset2.csv` in https://zenodo.org/record/7401275.

`datasets/k50_1_2_processed.tsv` - processed Mega dataset from `notebooks/01_processing.ipynb`.

`datasets/k50_1_2_processed_single.tsv` - only single-point mutations from `datasets/k50_1_2_processed.tsv`. This dataset was used to derive train, test and holdout sets.

### Protein sequences, BLAST

`fasta_blast/mega_seqs.fasta` - unique sequences of wild-type proteins (column _wt_seq_ in `datasets/k50_1_2_processed.tsv`, reconstructed from _mut_type_ and _aa_seq_ columns in `notebooks/01_processing.ipynb`).

All-against-all BLAST of Mega dataset:

```bash
makeblastdb -in mega_seqs.fasta -title 'wt' -dbtype prot
blastp -query mega_seqs.fasta -out blast_mega_against_mega.out -db mega_seqs.fasta -evalue 0.00001 -outfmt "6 qacc sacc qseq sseq pident length mismatch gapopen qstart qend sstart send evalue bitscore"
```

`fasta_blast/blast_mega_against_mega.tsv` - processed blast_mega_against_mega.out (sorted by sequence identity, removed blast of a sequence against the same sequence).

`fasta_blast/old_seqs.fasta` - sequences of proteins from "old" datasets: Myoglobin, S669, p53, S2648, Ssym.

`fasta_blast/S2648_seqs.fasta` - sequences of proteins from S2648.

`fasta_blast/blast_old_against_mega.out` - BLAST of proteins from "old" datasets against proteins from Mega dataset (BLAST and processing in `notebooks/03_blast.ipynb`).

`fasta_blast/blast_old_against_mega.out` - BLAST of proteins from "old" datasets against proteins from S2648 (BLAST and processing in `notebooks/03_blast.ipynb`).

### Train-test split

`notebooks/02_train_test_split.ipynb` - split of `datasets/k50_1_2_processed.tsv` into train and test by sequence identity threshold based on `blast_processed.tsv`.

`datasets/train_test_sets/cutoff_splits_diverse_train` - test and train splits from `notebooks/02_train_test_split.ipynb`. Dissimilar proteins in train.

`datasets/train_test_sets/cutoff_splits` - test and train splits from `notebooks/02_train_test_split.ipynb`. Dissimilar proteins in test.

### Datasets

In the `datasets` directory:

- `holdout.tsv` - Mega Holdout set
- `Myoglobin.tsv` - Kepp 2015
- `p53.tsv` - Pires et al., 2014
- `Ssym.tsv`  - Pucci et al., 2018
- `S669_420.tsv` - 420 out of 669 mutations of S669 dataset (Pancotti et al., 2022). Proteins similar to Mega datasets proteins were removed.
- `S669_411.tsv` - 411 out of 420 mutations of S669_420 dataset. Proteins similar to S2648 datasets proteins were removed.
- `S2648` - Dehouck et al., 2009
- `S2648_2441` - S2648 without mutations in proteins similar to Mega dataset.

Original datasets taken from ProDDG database (Marina A. Pak, Evgeny V. Kniazev, Igor D. Abramkin, Dmitry N. Ivankov. (2023). ProDDG - database of ∆∆G datasets and predictors. Available at: https://ivankovlab.ru/proddg). 

## Citation

If you used our work, please, cite:

```
Pak, M. A., Dovidchenko, N. V., Sharma, S. M., & Ivankov, D. N. (2023). New mega dataset combined with deep neural network makes a progress in predicting impact of mutation on protein stability. https://doi.org/10.1101/2022.12.31.522396
```
