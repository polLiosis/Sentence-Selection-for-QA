import re, random
from sklearn.utils import shuffle

def transform_to_collections(qids, questions, documents, old_questions, old_documents, offsets_start_list, offsets_end_list):
    ids, improved_queries, improved_collections, starting_queries, starting_collections, offsets_start, offsets_end = [], [], [], [], [], [], []
    prev_query = " "
    docs ,q, id, old_docs, old_q, os, oe = [], [], [], [], [], [], []
    for qid, query, documents, old_query, old_document, o_start, o_end, in zip(qids, questions, documents, old_questions, old_documents, offsets_start_list, offsets_end_list):
        if prev_query == query:
            id.append(qid)
            docs.append(document)
            q.append(query)
            old_q.append(old_query)
            old_docs.append(old_document)
            os.append(o_start)
            oe.append(o_end)
        else:
            if prev_query != " ":
                improved_queries.append(q)
                ids.append(id)
                improved_collections.append(docs)
                id, docs, q = [], [], []
                old_query.append(old_q)
                old_document.append(old_document)
                o_start.append(os)
                o_end.append(oe)
                old_q, old_docs, os, oe = [], [], [], []
            prev_query = query
            docs.append(document)
            q.append(query)
            id.append(qid)
            old_q.append(old_query)
            old_document.append(old_document)
            os.append(o_start)
            oe.append(o_end)
    ids.append(id)
    improved_queries.append(q)
    improved_collections.append(docs)
    starting_queries.append(old_q)
    starting_collections.append(old_docs)
    offsets_start.append(os)
    offsets_end.append(oe)
    return ids, improved_queries, improved_collections, starting_queries, starting_collections, offsets_start, offsets_end

def seperate_punctuation(dataset):
    with open(dataset, 'r') as dset:
        ids, questions, answers, old_questions, old_answers, starts, ends, doc_ids = [], [], [], [], [], [], [], []
        for line in dset:
            try:
                items = line[:-1].split("\t")
                qid = items[0]
                question = re.findall(r"[\w']+|[.,!?;']", items[1].lower())
                answer = re.findall(r"[\w']+|[.,!?;']", items[2].lower())
                old_q = items[1]
                old_a = items[2]
                start = items[3]
                end = items[4]
                did = items[5]

                ids.append(qid)
                questions.append(question)
                answers.append(answer)
                old_questions.append(old_q)
                old_answers.append(old_a)
                starts.append(start)
                ends.append(end)
                doc_ids.append(did)
            except:
                pass
    return ids, questions, answers, old_questions, old_answers, starts, ends, doc_ids


def clean_dataset(dataset):
    ids, questions, answers, old_questions, old_answers, starts, ends, doc_ids = seperate_punctuation(dataset)
    return ids, questions, answers, old_questions, old_answers, starts, ends, doc_ids


def improve_dataset(dataset):
    ids, questions_list, answers_list, old_questions_list, old_answers_list, starts, ends, doc_ids = clean_dataset(dataset)
    questions, answers, old_questions, old_answers = [], [], [], []
    for question in questions_list:
        question = " ".join(question)
        questions.append(question)
    for answer in answers_list:
        answer = " ".join(answer)
        answers.append(answer)
    for old_question in old_questions_list:
        old_question = "".join(old_question)
        old_questions.append(old_question)
    for old_answer in old_answers_list:
        old_answer = "".join(old_answer)
        old_answers.append(old_answer)

    temp_set = open(dataset, 'w')
    test_set = open(dataset, 'a')

    for id, question, answer, old_q, old_a, start, end, doc_id in zip(ids, questions, answers, old_questions, old_answers, starts, ends, doc_ids):
        test_set.write(id + "\t" + question + "\t" + answer + "\t" + old_q + "\t" + old_a + "\t" + str(start) + "\t" + str(end) + "\t" + str(doc_id) + "\n")



improve_dataset("Datasets/test.txt")




