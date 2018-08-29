## Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks

This is the implementation (keras) of a state-of-the-art model proposed by [Aliaksei Severyn and Alessandro Moschitti](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf).


#### Preprocessing
The first step is to parse the data. For the experiments three datasets where used (BioASQ, Trec QA, WikiQA), so we have to run parse_BiOASQ.py for BioASQ, parse_TrecQA for Trec QA or parse_WikiQA for WikiQA dataset.

#### Train & Test
Execute main.py. We have to call the appropriate function for each dataset.