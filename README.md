## Knowledge Graph Embedding for Factoid Entity Question Answering
KGE-FEQ is a hybrid framework designed to address the challenges of answering factoid entity questions. Unlike traditional open-domain question answering systems that rely on the explicit mention of entities in text or structured knowledge graphs constrained by predefined schemas, KGE-FEQ leverages a textual knowledge graph. This approach integrates structured knowledge from knowledge graphs with textual relationships derived from large text corpora. By embedding both the entities and their textual relationships, KGE-FEQ effectively retrieves relevant triples and identifies the most appropriate entity to answer a given question, circumventing the limitations of formal query languages and shallow ontologies.

KGE-FEQ operates through two main phases: Triple Retrieval and Answer Selection. In the Triple Retrieval phase, the model learns to retrieve a set of semantically related triples from a textual knowledge graph based on the question. Subsequently, during the Answer Selection phase, KGE-FEQ employs a knowledge graph embedding approach to score and rank entities within the retrieved triples. The model positions the embedding of the answer entity close to that of the question entity, taking into account the question, answer entities, and their textual relationships. Experimental evaluations demonstrate KGE-FEQ's superior performance across various benchmarks, outperforming existing methods in both open-domain and factoid entity question answering. 
## Files
The data, models, results and baselines for TriviaQA, SQuAD Open, NQ, QBLINk and QANTA can be downloaded [here](https://drive.google.com/drive/folders/1fQUyknhOIdm2N2O-xj8oSECcPixy9w4F?usp=share_link)

## DATA
The subset of TriviaQA, SQuAD Open, NQ, QBLink and QANTA dataset that is used to train and test our models are provided [here](https://drive.google.com/drive/folders/1fQUyknhOIdm2N2O-xj8oSECcPixy9w4F?usp=share_link). Download the data and place it under data folder.

## Pretrained Models
The pretrained models for both triple retrieval and answer selection, can be found [here](https://drive.google.com/drive/folders/1fQUyknhOIdm2N2O-xj8oSECcPixy9w4F?usp=share_link). Unzip and place it in model folder.

## Testing The Models
To test the models for the selected dataset, you can run the following code. Please make sure that pretrained models are downloaded and placed under the Model folder.

```
cd src/
python main.py test {dataset_name}
```

## Training The Models
To train model from scratch for the selected dataset, please run the following command. 
```
cd src/
python main.py train {dataset_name}
```

## Baseline Setting
To run the baselines we utilized the following Github repositories [Pygaggle](https://github.com/castorini/pygaggle/blob/master/docs/experiments-dpr-reader.md), [FID](https://github.com/facebookresearch/FiD) with the following settings.

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


