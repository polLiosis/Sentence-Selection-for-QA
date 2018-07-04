import re, random
from sklearn.utils import shuffle

def num_of_ones(labels):
    num = 0
    for label in labels:
        if label == 1:
            num += 1
    if num == 0:
        num = 1
    return num

def num_of_zeros(labels):
    num = 0
    for label in labels:
        if label == 0:
            num += 1
    if num == 0:
        num = 1
    return num

def positiveQueries(id, query, collection, label):
    num = num_of_ones(label)
    queries = []
    for i, q, col, l in zip(id, query, collection, label):
        if l == 1:
            queries.append(col)
    return num, queries

def negativeQueries(id, query, collection, label):
    num = num_of_ones(label)
    queries = []
    for i, q, col, l in zip(id, query, collection, label):
        if l == 0:
            queries.append(col)
    return num, queries

def non_overlap(col, positive_docs):
    overlap = True
    for doc in positive_docs:
        if col in doc:
            overlap = False
            break
    return overlap

def transform_to_collection(qids, questions, documents, y):
    ids, queries, collections, labels = [], [], [], []
    prev_query = " "
    docs, q, ys, id = [], [], [], []
    for qid, query, document, label in zip(qids, questions, documents, y):
        if prev_query == query:
            id.append(qid)
            docs.append(document)
            ys.append(label)
            q.append(query)
        else:
            if prev_query != " ":
                queries.append(q)
                ids.append(id)
                collections.append(docs)
                labels.append(ys)
                id, docs, q, ys = [], [], [], []
            prev_query = query
            docs.append(document)
            ys.append(label)
            q.append(query)
            id.append(qid)
    ids.append(id)
    queries.append(q)
    collections.append(docs)
    labels.append(ys)
    return ids, queries, collections, labels


def seperate_punctuation(dataset):
    with open( dataset, 'r') as dset:
        ids, questions, answers, labels = [], [], [], []
        for line in dset:
            try:
                items = line[:-1].split("\t")

                qid = items[0]
                question = re.findall(r"[\w']+|[.,!?;']", items[1].lower())
                answer = re.findall(r"[\w']+|[.,!?;']", items[2].lower())
                label = int(items[3])

                ids.append(qid)
                questions.append(question)
                answers.append(answer)
                labels.append(label)
            except:
                pass
    return ids, questions, answers, labels


def balance_dataset(ids, questions, answers, labels):
    ids, questions, answers, labels = transform_to_collection(ids, questions, answers, labels)
    temp_ids, temp_questions, temp_answers, temp_labels = [], [], [], []
    for id, query, collection, label in zip(ids, questions, answers, labels):
        ones, zeros = [], []
        id, query, collection, label = shuffle(id, query, collection, label)
        positive_num, positive_docs = positiveQueries(id, query, collection, label)
        negative_num, negative_docs = negativeQueries(id, query, collection, label)
        zeros_added = 0
        for i, q, col, l in zip(id, query, collection, label):
            if l == 1 and col not in ones:
                temp_ids.append(i)
                temp_questions.append(q)
                temp_answers.append(col)
                temp_labels.append(l)
                ones.append(col)
            elif l == 0 and col not in positive_docs and col not in zeros:
                if non_overlap(col, positive_docs):
                    if zeros_added < positive_num:
                        temp_ids.append(i)
                        temp_questions.append(q)
                        temp_answers.append(col)
                        temp_labels.append(l)
                        zeros_added +=1
                        zeros.append(col)
    return temp_ids, temp_questions, temp_answers, temp_labels


def clean_dataset(dataset):
    ids, questions, answers, labels = seperate_punctuation(dataset)
    return ids, questions, answers, labels


def improve_dataset(dataset):
    ids, questions_list, answers_list, labels = clean_dataset(dataset)
    questions, answers = [], []
    for question in questions_list:
        question = " ".join(question)
        questions.append(question)
    for answer in answers_list:
        answer = " ".join(answer)
        answers.append(answer)

    # We balance the dataset
    ids, questions, answers, labels = balance_dataset(ids, questions, answers, labels)

    temp_set = open("Datasets/train.txt", 'w')
    train_set = open("Datasets/train.txt", 'a')
    for id, question, answer, label in zip(ids, questions, answers, labels):
        train_set.write(id + "\t" + question + "\t" +answer + "\t" + str(label) + "\n")



improve_dataset("Datasets/train.txt")