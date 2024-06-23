The data, models, results and baselines for TriviaQA, SQuAD Open, NQ, QBLINk and QANTA can be downloaded [here](https://drive.google.com/drive/folders/1fQUyknhOIdm2N2O-xj8oSECcPixy9w4F?usp=share_link)

## DATA
The subset of TriviaQA, SQuAD Open, NQ, QBLink and QANTA dataset that is used to train and test our models are provided [here](https://drive.google.com/drive/folders/1fQUyknhOIdm2N2O-xj8oSECcPixy9w4F?usp=share_link). Download the data and place it under data folder.

## Pretrained Models
The pretrained models for both triple retrieval and answer selection, can be found [here](https://drive.google.com/drive/folders/1fQUyknhOIdm2N2O-xj8oSECcPixy9w4F?usp=share_link). Unzip and place it in model folder.

## Testing The Models
To test the models for the selected dataset run the following code. Make sure that pretrained models are downloaded and placed under the Model folder.



## Training The Models

## Baseline Setting
To run the baselines we utilized the following settings.

|DATASET| Best-k|Retriver| Reader | Beta | Gamma |
|--------|--------|--------|--------|--------|--------|
| NQ | 50  | DPR  | Single  | -  | -  |
| NQ | 5  | DPR+BM25  | Single  | -  | -  | 
| NQ | 50  | DPR F  |  Single  | 1.0  | 0.55  |
| NQ | 200  | DPR F+BM25  | Single  | 1.0  | 0.63  |
| NQ | 50  | GAR  | Single  | -  | -  |
| NQ | 50  | GAR+BM25  | Single  | -  | -  |
| NQ | 50  | GAR F  | Single  | 0.43  | 0.30  |
| NQ | 50  | GAR F+BM25  | Single  | 0.32  | 0.19  |
| TriviaQA | 500  | DPR  | Multiset  | -  | -  |
| TriviaQA | 100  | DPR+BM25  | Multiset  | -  | -  | 
| TriviaQA | 500  | DPR F  |  Multiset  | 1.0  | 0.18  |
| TriviaQA | 480  | DPR F+BM25  | Multiset  | 1.0  | 0.17  |
| TriviaQA | 480  | GAR  | Multiset  | -  | -  |
| TriviaQA | 50  | GAR+BM25  | Multiset  | -  | -  |
| TriviaQA | 500  | GAR F  | Multiset  | 1.18  | 0.24  |
| TriviaQA | 480  | GAR F+BM25  | Multiset  | 0.76  | 0.15  |
| SQuAD | 100  | DPR  | Single  | -  | -  |
| SQuAD | 5  | DPR+BM25  | Single  | -  | -  | 
| SQuAD | 100  | DPR F  |  Single  | 1.0  | 0.55  |
| SQuAD | 5  | DPR F+BM25  | Single  | 1.0  | 0.63  |
| SQuAD | 100  | GAR  | Single  | -  | -  |
| SQuAD | 5  | GAR+BM25  | Single  | -  | -  |
| SQuAD | 100  | GAR F  | Single  | 0.43  | 0.30  |
| SQuAD | 20  | GAR F+BM25  | Single  | 0.32  | 0.19  |


