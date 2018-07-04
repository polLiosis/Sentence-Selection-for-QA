import nltk
from random import shuffle

def getOffsets(abstract, snippet):
    en_abstract = bytes(abstract, encoding = "utf-8")
    en_snippet = bytes(snippet, encoding = "utf-8")
    start = en_abstract.find(en_snippet)
    return start, start + len(en_snippet)

def createTrainSetForm(qid, question, docid, snippet):
    rel_snip = []
    rel_snip.append(qid + "\t" + question + "\t" + snippet + "\t" + "1")
    QA_pairs = []
    new_set = open("Datasets/train.txt", 'a')
    with open("PubMeds/Abstracts/" + docid + "/" + docid + "_abstract.txt", 'r') as f:
        abstract = f.read()
        # We remove the current snippet from the abstract
        abstract = abstract.replace(snippet, ' ')
        sentences = nltk.sent_tokenize(abstract)
        for sentence in sentences:
            if sentence != "." or sentence != " . ":
                QA_pairs.append(qid + "\t" + question + "\t" + sentence + "\t" + "0")

        # Shuffle the irrelevant sentences and then we pick 5 random ones together with the relevant snippet
        # We use this technique in order to reduce the number of irrelevant snippets
        shuffle(QA_pairs)
        QA_pairs = QA_pairs[:5]
        # Concatenate the two lists
        final_QA_pairs = set(rel_snip + QA_pairs)

        for pair in final_QA_pairs:
            try:
                new_set.write(pair + "\n")
            except:
                print("Error for QA pair: ", pair)
        f.close()
        new_set.close()

def createTestSetForm(qid, question, docid):
    new_set = open("Datasets/test.txt", 'a')
    with open("PubMeds/Abstracts/" + docid + "/" + docid + "_abstract.txt", 'r') as f:
        QA_pairs = []
        abstract = f.read()
        sentences = nltk.sent_tokenize(abstract)
        for sentence in sentences:
            if sentence != "." or sentence != " . ":
                offset_start, offset_end = getOffsets(abstract, sentence)
                QA_pairs.append(qid + "\t" + question + "\t" + sentence + "\t" + str(offset_start) + "\t" + str(offset_end) + "\t" + str(docid))

        for pair in QA_pairs:
            try:
                new_set.write(pair + "\n")
            except:
                print("Error for QA pair: ", pair)
        f.close()
        new_set.close()




