### Implementation of BM25 ranking function

Okapi BM25 is a ranking function used by search engines to rank documents ac-
cording to their relevance to a given search query and it is based on the probabilistic retrieval framework.

In our case we have a number of questions and for each question we have a collection of documents/snippets. The objective is for each question to select the top ranked snippets (using BM25 score).