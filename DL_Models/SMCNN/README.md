## Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks

This is the implementation (keras) of a state-of-the-art model proposed by [Aliaksei Severyn and Alessandro Moschitti](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf).

#### Specification
- **BioASQ_dataset / TrecQA / WikiQA**: Folders with BioASQ, TrecQA and WikiQA dataset respectively.
- **BM25 / BM25_files**: Folders where we save BM25 scores etc.
- **Embeddings**: Folder with pre-trained embeddings.
- **experiments**: Here we save the results of our experiments.
- **parse_BioASQ.py / parse_TrecQA.py / parse_WikiQA.py**: scripts used for parsing of the respective datasets.
- **BM25.py**: script used in order to compute BM25 scores.
- **SMCNN_model.py**: Implementation of the specific model.
- **main.py**: Script used in order to train and test the model.


#### Preprocessing
The first step is to parse the data. For the experiments three datasets where used (BioASQ, Trec QA, WikiQA), so we have to run parse_BiOASQ.py for BioASQ, parse_TrecQA for Trec QA or parse_WikiQA for WikiQA dataset.

#### Train & Test
Execute main.py. We have to call the appropriate function for each dataset.