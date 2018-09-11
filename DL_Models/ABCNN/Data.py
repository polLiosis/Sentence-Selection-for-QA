import pickle

# Load df and idf scores
def load_idf_scores():
    with open('BM25_files/idf.pkl', 'rb') as f:
        idf_scores = pickle.load(f)
    with open('BM25_files/df.pkl', 'rb') as f:
        df_scores = pickle.load(f)
    return df_scores, idf_scores

# Load BioASQ dataset
def load_dataset(dataset):
    qids, questions, answers, labels = [], [], [], []
    with open(dataset, 'r') as file:
        for line in file:
            items = line[:-1].split("\t")
            qid = items[0]
            question = items[1].lower().split()
            answer = items[2].lower().split()
            qids.append(qid)
            questions.append(question)
            answers.append(answer)
    return qids, questions, answers, labels

# Return three lists:
# queries -> [ [q1], [q2], ..., [qn]]
# collections -> Contains snippets for each query [ [q1s1, q1s2...], ..., [qns1...] ]
# labels -> Contains the labels for each snippet of collections list [ [1,0...] , ..., [1...]]
def transform_to_collections(questions, documents, y):
    queries, collections, labels = [], [], []
    prev_query = " "
    docs = []
    ys = []
    for query, document, label in zip(questions, documents, y):
        if prev_query == query:
            docs.append(documents)
            ys.append(label)
        else:
            queries.append(query)
            if prev_query != " ":
                collections.append(docs)
                labels.append(ys)
                docs, ys = [], []
            prev_query = query
            docs.append(document)
            ys.append(label)
    collections.append(docs)
    labels.append(ys)
    print(queries)
    return queries, collections, labels
