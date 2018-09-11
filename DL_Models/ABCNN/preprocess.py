import numpy as np
import nltk, gensim, BM25, itertools, pickle

# Retrieve the embdeddings from bin file
class Word2Vec():
    def __init__(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./Embeddings/embeddings.bin', binary=True)

        # Create a random vector, in case we cannot find the word embedding in our pre-trained embeddings
        self.unknowns = np.random.uniform(-0.01, 0.01, 200).astype("float32")

    # Get the embedding vector for a specific word
    def get(self, word):
        if word not in self.model.vocab:
            return self.unknowns
        else:
            return self.model.word_vec(word)


class Data():
    #initialize basic variables
    def __init__(self, word2vec, max_len=0):
        # sentence1, sentence2, labels, features
        self.s1s, self.s2s, self.labels, self.features = [], [], [], []
        self.qid, self.old_answer, self.old_question, self.start, self.end, self.did = [], [], [], [], [], []
        self.index, self.max_len, self.word2vec = 0, max_len, word2vec

    # open specific file
    def open_file(self):
        pass

    # Check if we have available data
    def is_available(self):
        if self.index < self.data_size:
            return True
        else:
            return False

    # Reset index to zero (0)
    def reset_index(self):
        self.index = 0

    def next(self):
        if (self.is_available()):
            self.index += 1
            return self.data[self.index - 1]
        else:
            return

    # Retrieve next batch
    def next_batch(self, batch_size):
        batch_size = min(self.data_size - self.index, batch_size)
        s1_mats, s2_mats = [], []

        for i in range(batch_size):
            s1 = self.s1s[self.index + i]
            s2 = self.s2s[self.index + i]

            # [1, d0, s]
            s1_mats.append(np.expand_dims(np.pad(np.column_stack([self.word2vec.get(w) for w in s1]),
                                                 [[0, 0], [0, self.max_len - len(s1)]],
                                                 "constant"), axis=0))

            s2_mats.append(np.expand_dims(np.pad(np.column_stack([self.word2vec.get(w) for w in s2]),
                                                 [[0, 0], [0, self.max_len - len(s2)]],
                                                 "constant"), axis=0))

        # [batch_size, d0, s]
        batch_s1s = np.concatenate(s1_mats, axis=0)
        batch_s2s = np.concatenate(s2_mats, axis=0)
        batch_labels = self.labels[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size

        return batch_s1s, batch_s2s, batch_labels, batch_features

    def getMoreInfo(self):
        return self.qid, self.old_question, self.old_answer, self.start, self.end, self.did


class BioASQ(Data):
    # Load df and idf scores
    def load_idf_scores(self):
        with open('BM25_files/idf.pkl', 'rb') as f:
            idf_scores = pickle.load(f)
        with open('BM25_files/df.pkl', 'rb') as f:
            df_scores = pickle.load(f)
        return df_scores, idf_scores

    # Load BioASQ dataset
    def load_dataset(self, dataset):
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

    def compute_Overlaps(self, q_tokens, d_tokens, q_idf):
        # Map term to idf before set() change the term order
        q_terms_idf = {}
        for i in range(len(q_tokens)):
            q_terms_idf[q_tokens[i]] = q_idf[i]

        # Query Uni and Bi gram sets
        query_uni_set = set()
        query_bi_set = set()
        for i in range(len(q_tokens) - 1):
            query_uni_set.add(q_tokens[i])
            query_bi_set.add((q_tokens[i], q_tokens[i + 1]))
        query_uni_set.add(q_tokens[-1])

        # Doc Uni and Bi gram sets
        doc_uni_set = set()
        doc_bi_set = set()
        for i in range(len(d_tokens) - 1):
            doc_uni_set.add(d_tokens[i])
            doc_bi_set.add((d_tokens[i], d_tokens[i + 1]))
        doc_uni_set.add(d_tokens[-1])

        unigram_overlap = 0
        idf_uni_overlap = 0
        idf_uni_sum = 0
        for ug in query_uni_set:
            if ug in doc_uni_set:
                unigram_overlap += 1
                idf_uni_overlap += q_terms_idf[ug]
            idf_uni_sum += q_terms_idf[ug]
        unigram_overlap /= len(query_uni_set)
        idf_uni_overlap /= idf_uni_sum

        bigram_overlap = 0
        for bg in query_bi_set:
            if bg in doc_bi_set:
                bigram_overlap += 1
        bigram_overlap /= len(query_bi_set)

        return unigram_overlap, bigram_overlap, idf_uni_overlap

    def open_file(self, mode):
        df_scores, idf_scores = self.load_idf_scores()
        documents = []
        with open("./BioASQ_Corpus/BioASQ-" + mode + ".txt", "r", encoding="utf-8") as f:
            for line1, line2 in itertools.zip_longest(*[f] * 2):
                items = line1[:-1].split("\t")
                answer = items[2].lower().split()
                documents.append(answer)
        avgdl = BM25.compute_avgdl(documents)

        documents = []
        with open("./BioASQ_Corpus/BioASQ-train.txt", "r", encoding="utf-8") as f:
            for line1, line2 in itertools.zip_longest(*[f] * 2):
                items = line1[:-1].split("\t")
                answer = items[2].lower().split()
                documents.append(answer)
        train_avgdl = BM25.compute_avgdl(documents)

        # Compute mean and deviation for Z-score normalization
        maxim = max(idf_scores.keys(), key=(lambda i: idf_scores[i]))
        rare_word_value = idf_scores[maxim]
        mean, deviation = BM25.compute_Zscore_values("./BioASQ_Corpus/BioASQ-train.txt", idf_scores, train_avgdl, 1.2, 0.75, rare_word_value)

        print("mean", mean)
        print("deviation", deviation)

        with open("./BioASQ_Corpus/BioASQ-" + mode + ".txt", "r", encoding="utf-8") as f:
            # We retrieve the stopwords of the english language from the nltk library
            stopwords = nltk.corpus.stopwords.words("english")

            if ((mode == 'test') or (mode == 'dev')):
                for line in f:
                    items = line[:-1].split("\t")
                    s1 = items[1].lower().split()
                    # truncate answers to 40 tokens.
                    s2 = items[2].lower().split()[:40]
                    label = int(0)

                    qid = items[0]
                    old_question = items[3]
                    old_answer = items[4]
                    start = items[5]
                    end = items[6]
                    did = items[7]

                    self.s1s.append(s1)
                    self.s2s.append(s2)
                    self.labels.append(label)

                    self.qid.append(qid)
                    self.old_question.append(old_question)
                    self.old_answer.append(old_answer)
                    self.start.append(start)
                    self.end.append(end)
                    self.did.append(did)

                    BM25score = BM25.similarity_score(s1, s2, 1.2, 0.75, idf_scores, avgdl, True, mean, deviation, rare_word_value)

                    q_idfs = []
                    for token in s1:
                        try:
                            q_idfs.append(idf_scores[token])
                        except:
                            q_idfs.append(1)
                    unigram_overlap, bigram_overlap, idf_uni_overlap = self.compute_Overlaps(s1, s2, q_idfs)

                    word_cnt = len([word for word in s1 if (word not in stopwords) and (word in s2)])

                    self.features.append([len(s1), len(s2), word_cnt, BM25score, unigram_overlap, bigram_overlap, idf_uni_overlap])

                    local_max_len = max(len(s1), len(s2))
                    if local_max_len > self.max_len:
                        self.max_len = local_max_len
            elif (mode == 'train'):
                for line in f:
                    items = line[:-1].split("\t")
                    # We retieve the question, the candidate answer and the label from each line
                    s1 = items[1].lower().split()
                    # truncate answers to 40 tokens.
                    s2 = items[2].lower().split()[:40]
                    label = int(items[3])

                    self.s1s.append(s1)
                    self.s2s.append(s2)
                    self.labels.append(label)

                    word_cnt = len([word for word in s1 if (word not in stopwords) and (word in s2)])

                    BM25score = BM25.similarity_score(s1, s2, 1.2, 0.75, idf_scores, avgdl, True, mean, deviation, rare_word_value)
                    q_idfs = []
                    for token in s1:
                        try:
                            q_idfs.append(idf_scores[token])
                        except:
                            q_idfs.append(1)
                    unigram_overlap, bigram_overlap, idf_uni_overlap = self.compute_Overlaps(s1, s2, q_idfs)

                    self.features.append([len(s1), len(s2), word_cnt, BM25score, unigram_overlap, bigram_overlap, idf_uni_overlap])

                    # Get the maximum length of a sentence, while reading the train/test file
                    local_max_len = max(len(s1), len(s2))
                    if local_max_len > self.max_len:
                        self.max_len = local_max_len

        self.data_size = len(self.s1s)
        flatten = lambda l: [item for sublist in l for item in sublist]
        q_vocab = list(set(flatten(self.s1s)))
        idf = {}
        for w in q_vocab:
            idf[w] = np.log(self.data_size / len([1 for s1 in self.s1s if w in s1]))
        for i in range(self.data_size):
            wgt_word_cnt = sum([idf[word] for word in self.s1s[i] if (word not in stopwords) and (word in self.s2s[i])])
            self.features[i].append(wgt_word_cnt)
        self.num_features = len(self.features[0])
