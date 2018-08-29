import numpy, json, time
import keras.backend as K
from collections import defaultdict
from keras.models import model_from_json
from SMCNN_model import load_embeddings, SMCNN
from keras.callbacks import Callback
from operator import itemgetter


UNKNOWN_WORD_IDX = 0

Q_DEV = []
A_DEV = []
BM_DEV = []
Y_DEV = []
IDS_DEV = []

Q_TRAIN = []
A_TRAIN = []
BM_TRAIN = []
Y_TRAIN = []
QIDS_TRAIN = []

qids_test, q_test, a_test, y_true, y_pred, old_questions_test, old_answers_test, starts_test, ends_test, BM25_test, y_test, dids_test = [], [], [], [], [], [], [], [], [], [], [], []

model = None

global l


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

# Return MAP score for train set
class MAP_train(Callback):
    maps = []
    def compute_map(self):
        # Create and fill a dictionary : dict[qid]->(prediction, label)
        print('Q_TRAIN', Q_TRAIN.shape)
        print('A_TRAIN', A_TRAIN.shape)
        x_val, y_true = [Q_TRAIN, A_TRAIN, BM_TRAIN], Y_TRAIN
        y_pred = self.model.predict(x_val)

        qid2cand = defaultdict(list)
        for qid, label, pred in zip(QIDS_TRAIN, y_true, y_pred):
            qid2cand[qid].append((pred, label))

        # We need to save the average precisions
        average_precs = []
        # For a specific qid we get the predictions and the correct labels
        for qid, candidates in qid2cand.items():
            average_prec = 0
            running_correct_count = 0
            # we sort the data for the loop
            for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
                # If the label is 1 we have a correct candidate answer
                if label > 0:
                    running_correct_count += 1
                    average_prec += float(running_correct_count) / i  # Compute AP
            average_precs.append(average_prec / (running_correct_count + 1e-6))
        map_score = sum(average_precs) / len(average_precs)
        return map_score

    def on_epoch_end(self, epoch, logs={}):
        score = self.compute_map()
        print()
        print("\n (TRAIN) MAP for epoch %d is %f" % (epoch + 1, score))
        f = open("experiments/MAP_train_results.txt", "a")
        f.write(str(score) + "\n")
        f.close()
        self.maps.append(score)

# Return MRR score for ytrain set
class MRR_train(Callback):

    mrrs = []
    def compute_mrr(self):
        # Create and fill a dictionary : dict[qid]->(prediction, label)
        x_val, y_true = [Q_TRAIN, A_TRAIN, BM_TRAIN], Y_TRAIN
        y_pred = self.model.predict(x_val)
        qid2cand = defaultdict(list)

        # Fill the directory
        for qid, label, pred in zip(QIDS_TRAIN, y_true, y_pred):
            qid2cand[qid].append((pred, label))

        mrr_scores = []
        # for each query
        for qid, candidates in qid2cand.items():
            mrr_score = 0
            # For each document retrieved
            for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
                # If the label is 1 we have a correct candidate answer
                if label > 0:
                    mrr_score = float(1)/i
                    break
            # We save MRR score for the specific query
            mrr_scores.append(mrr_score)
        # Return the sum of the calculated MRR scores devided by the number of the queries
        return sum(mrr_scores)/len(mrr_scores)

    def on_epoch_end(self, epoch, logs={}):
        score = self.compute_mrr()
        print("\n(TRAIN)MRR for epoch %d is %f" % (epoch + 1, score))
        print()
        f = open("experiments/MRR_train_results.txt", "a")
        f.write(str(score) + "\n")
        f.close()
        self.mrrs.append(score)

# Returns MAP score for dev set
class MAP_dev(Callback):
    maps = []

    def compute_map(self, epoch):
        # Create and fill a dictionary : dict[qid]->(prediction, label)
        print('Q_DEV', Q_DEV.shape)
        print('A_DEV', A_DEV.shape)
        x_val, y_true = [Q_DEV, A_DEV, BM_DEV], Y_DEV
        y_pred = self.model.predict(x_val)

        qid2cand = defaultdict(list)
        for qid, label, pred in zip(QIDS_DEV, y_true, y_pred):
            qid2cand[qid].append((pred, label))

        # We need to save the average precisions
        average_precs = []
        # For a specific qid we get the predictions and the correct labels
        for qid, candidates in qid2cand.items():
            average_prec = 0
            running_correct_count = 0
            # we sort the data for the loop
            for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
                # If the label is 1 we have a correct candidate answer
                if label > 0:
                    running_correct_count += 1
                    average_prec += float(running_correct_count) / i  # Compute AP
            average_precs.append(average_prec / (running_correct_count + 1e-6))  # Save the currect average precision
        map_score = sum(average_precs) / len(average_precs)


        return map_score

    def on_epoch_end(self, epoch, logs={}):
        score = self.compute_map(epoch + 1)
        print()
        print("\nMAP for epoch %d is %f" % (epoch + 1, score))
        f = open("experiments/MAP_dev_results.txt", "a")
        f.write(str(score) + "\n")
        f.close()
        self.maps.append(score)

# Return MRR score for dev set
class MRR_dev(Callback):

    mrrs = []

    def compute_mrr(self):
        # Create and fill a dictionary : dict[qid]->(prediction, label)
        x_val, y_true = [Q_DEV, A_DEV, BM_DEV], Y_DEV
        y_pred = self.model.predict(x_val)
        qid2cand = defaultdict(list)

        # Fill the directory
        for qid, label, pred in zip(QIDS_DEV, y_true, y_pred):
            qid2cand[qid].append((pred, label))

        mrr_scores = []
        # for each query
        for qid, candidates in qid2cand.items():
            mrr_score = 0
            # For each document retrieved
            for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
                # If the label is 1 we have a correct candidate answer
                if label > 0:
                    mrr_score = float(1)/i
                    break
            # We save MRR score for the specific query
            mrr_scores.append(mrr_score)
        # Return the sum of the calculated MRR scores devided by the number of the queries
        return sum(mrr_scores)/len(mrr_scores)

    def on_epoch_end(self, epoch, logs={}):
        score = self.compute_mrr()
        print("\nMRR for epoch %d is %f" % (epoch + 1, score))
        print()
        f = open("experiments/MRR_dev_results.txt", "a")
        f.write(str(score) + "\n")
        f.close()
        self.mrrs.append(score)

# Save Model for the specific epoch
class save_Model_state(Callback):
    def on_epoch_end(self, epoch, logs={}):
        # serialize model to JSON
        print("Saving model for epoch " + str(epoch + 1))
        model_json = model.to_json()
        with open("Model_files/model_"+ str(epoch + 1) + ".json", "w") as json_file:
            json_file.write(model_json)
        model_yaml = model.to_yaml()
        with open("Model_files/model_"+ str(epoch + 1) + ".yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights("Model_files/model_"+ str(epoch + 1) + ".h5")
        print("Model was saved with success!!!")


def MAP_MRR_score(qids_test, q_test, a_test, y_test, BM25_test):
    # Create and fill a dictionary : dict[qid]->(prediction, label)
    x_val, y_true = [q_test, a_test, BM25_test], y_test
    #x_val, y_true = [q_test, a_test], y_test
    y_pred = model.predict(x_val)
    # createPredFile(qids_test, x_val, y_pred, y_true, q_test, a_test)

    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids_test, y_true, y_pred):
        qid2cand[qid].append((pred, label))

    # We need to save the average precisions
    average_precs = []
    # For a specific qid we get the predictions and the correct labels
    for qid, candidates in qid2cand.items():
        average_prec = 0
        running_correct_count = 0
        # we sort the data for the loop
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            # If the label is 1 we have a correct candidate answer
            if label > 0:
                running_correct_count += 1
                average_prec += float(running_correct_count) / i  # Compute AP
        average_precs.append(average_prec / (running_correct_count  + 1e-6))  # Save the currect average precision
    map_score = sum(average_precs) / len(average_precs)

    mrr_scores = []
    # for each query
    for qid, candidates in qid2cand.items():
        mrr_score = 0
        # For each document retrieved
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            # If the label is 1 we have a correct candidate answer
            if label > 0:
                mrr_score = float(1) / i
                break
        # We save MRR score for the specific query
        mrr_scores.append(mrr_score)
    # Return the sum of the calculated MRR scores devided by the number of the queries
    mrr_score = sum(mrr_scores) / len(mrr_scores)


    return map_score, mrr_score

class evaluate_test(Callback):
    def evaluate(self, epoch):
        x_val, y_true = [q_test, a_test, BM25_test], y_test
        y_pred = model.predict(x_val)

        f = open('vocab.json', 'r')
        vocab = json.load(f)
        qid2cand = defaultdict(list)
        dict2key = defaultdict(list)
        for qid, question, answer, label, pred, old_q, old_a, old_os, old_oe in zip(qids_test, q_test, a_test, y_true,
                                                                                    y_pred, old_questions_test,
                                                                                    old_answers_test, starts_test,
                                                                                    ends_test):
            qid2cand[qid].append((question, answer, pred, old_q, old_a, old_os, old_oe))
        for qid, question, answer, label, pred, old_q, old_a, old_os, old_oe, doc_id in zip(qids_test, q_test, a_test,
                                                                                            y_true, y_pred,
                                                                                            old_questions_test,
                                                                                            old_answers_test,
                                                                                            starts_test, ends_test,
                                                                                            dids_test):
            dict2key[qid, old_q].append((answer, pred, old_q, old_a, old_os, old_oe, doc_id))

        print("Creating json file for BioASQ evaluation!!!")
        createJsonFile(dict2key, vocab, epoch)

    def on_epoch_end(self, epoch, logs={}):
        self.evaluate(epoch + 1)

# Train set using BioASQ dataset
def train_on_BioASQ():

    # Load BioASQ data
    print("Loading BioASQ data from external/numpy files")

    # Load train set
    qids_train = numpy.load("BioASQ_dataset/train_qids.npy")
    q_train = numpy.load("BioASQ_dataset/train_questions.npy")
    a_train = numpy.load("BioASQ_dataset/train_answers.npy")
    y_train = numpy.load("BioASQ_dataset/train_labels.npy")
    BM25_train = numpy.load("BioASQ_dataset/train_BM25scores.npy")

    # Load development set
    qids_dev = numpy.load("BioASQ_dataset/dev_qids.npy")
    q_dev = numpy.load("BioASQ_dataset/dev_questions.npy")
    a_dev = numpy.load("BioASQ_dataset/dev_answers.npy")
    y_dev = numpy.load("BioASQ_dataset/dev_labels.npy")
    BM25_dev = numpy.load("BioASQ_dataset/dev_BM25scores.npy")

    global qids_test, q_test, a_test, y_true, y_pred, old_questions_test, old_answers_test, starts_test, ends_test, BM25_test, y_test, dids_test

    # Load test set
    qids_test = numpy.load("BioASQ_dataset/test_qids.npy")
    q_test = numpy.load("BioASQ_dataset/test_questions.npy")
    a_test = numpy.load("BioASQ_dataset/test_answers.npy")
    y_test = numpy.load("BioASQ_dataset/test_labels.npy")
    BM25_test = numpy.load("BioASQ_dataset/test_BM25scores.npy")
    old_questions_test = numpy.load("BioASQ_dataset/test_old_questions.npy")
    old_answers_test = numpy.load("BioASQ_dataset/test_old_answers.npy")
    starts_test = numpy.load("BioASQ_dataset/test_starts.npy")
    ends_test = numpy.load("BioASQ_dataset/test_ends.npy")
    dids_test = numpy.load("BioASQ_dataset/test_doc_ids.npy")

    # Load max lens
    print("q_train shape: ", q_train.shape)
    print("a_train shape: ", a_train.shape)
    print("BM25_train shape: ", BM25_train.shape)
    max_ques_len = q_train.shape[1]
    max_ans_len = a_train.shape[1]
    print("maximum question len: ", max_ques_len)
    print("maximum answer len: ", max_ans_len)
    print(a_test.shape)

    global model
    model = SMCNN(max_ques_len, max_ans_len)

    global Q_DEV, A_DEV, Y_DEV, QIDS_DEV, BM_DEV, PSNIPS_DEV, RSNIPS_DEV, Q_TRAIN, A_TRAIN, BM_TRAIN, Y_TRAIN, QIDS_TRAIN
    Q_DEV = q_dev
    A_DEV = a_dev
    Y_DEV = y_dev
    BM_DEV = BM25_dev
    QIDS_DEV = qids_dev

    Q_TRAIN = q_train
    A_TRAIN = a_train
    BM_TRAIN = BM25_train
    Y_TRAIN  = y_train
    QIDS_TRAIN = qids_train

    map_dev_callback = MAP_dev()
    mrr_dev_callback = MRR_dev()
    map_train_callback = MAP_train()
    mrr_train_callback = MRR_train()
    save_model = save_Model_state()
    test_eval = evaluate_test()
    history = model.fit([q_train, a_train, BM25_train],
                        y_train, batch_size=200, nb_epoch=2,
                        validation_data=([q_dev, a_dev, BM25_dev], y_dev),
                        callbacks=[map_dev_callback, mrr_dev_callback, save_model, test_eval])

    print(history.history.keys())


def train_on_TrecQA():

    # Load TrecQA data
    print("Loading TrecQA data from external/numpy files")

    # Load train set
    qids_train = numpy.load("TrecQA/train_qids.npy")
    q_train = numpy.load("TrecQA/train_questions.npy")
    a_train = numpy.load("TrecQA/train_answers.npy")
    y_train = numpy.load("TrecQA/train_labels.npy")
    BM25_train = numpy.load("TrecQA/train_BM25scores.npy")

    # Load development set
    qids_dev = numpy.load("TrecQA/dev_qids.npy")
    q_dev = numpy.load("TrecQA/dev_questions.npy")
    a_dev = numpy.load("TrecQA/dev_answers.npy")
    y_dev = numpy.load("TrecQA/dev_labels.npy")
    BM25_dev = numpy.load("TrecQA/dev_BM25scores.npy")

    # Load vocabulary and max lens
    f = open('vocab.json', 'r')
    vocab = json.load(f)
    print("q_train shape: ", q_train.shape)
    print("a_train shape: ", a_train.shape)
    print("BM25_train shape: ", BM25_train.shape)
    max_ques_len = q_train.shape[1]
    max_ans_len = a_train.shape[1]
    print("maximum question len: ", max_ques_len)
    print("maximum answer len: ", max_ans_len)

    # Load embeddings
    # embedding, dim = load_embeddings('./Embeddings/pubmed_s100w10_min.bin', vocab)
    embedding, dim = load_embeddings('./Embeddings/embeddings.bin', vocab)
    print('dim: ', dim)
    print("emdedding matrix shape: ", embedding.shape)

    # Call Model
    global model
    model = SMCNN(max_ques_len, max_ans_len)
    global Q_DEV, A_DEV, Y_DEV, QIDS_DEV, BM_DEV, Q_TRAIN, A_TRAIN, BM_TRAIN, Y_TRAIN, QIDS_TRAIN
    Q_DEV = q_dev
    A_DEV = a_dev
    Y_DEV = y_dev
    BM_DEV = BM25_dev
    QIDS_DEV = qids_dev
    Q_TRAIN = q_train
    A_TRAIN = a_train
    BM_TRAIN = BM25_train
    Y_TRAIN = y_train
    QIDS_TRAIN = qids_train

    map_dev_callback = MAP_dev()
    mrr_dev_callback = MRR_dev()
    map_train_callback = MAP_train()
    mrr_train_callback = MRR_train()
    save_model = save_Model_state()

    history = model.fit([q_train, a_train, BM25_train],
                        y_train, batch_size=200, nb_epoch=3,
                        validation_data=([q_dev, a_dev, BM25_dev], y_dev),
                        callbacks=[map_dev_callback, mrr_dev_callback, map_train_callback, mrr_train_callback, save_model])

    # list all data in history
    print(history.history.keys())

def train_on_WikiQA():
    # Load TrecQA data
    print("Loading WikiQA data from external/numpy files")

    # Load train set
    qids_train = numpy.load("WikiQA/train_qids.npy")
    q_train = numpy.load("WikiQA/train_questions.npy")
    a_train = numpy.load("WikiQA/train_answers.npy")
    y_train = numpy.load("WikiQA/train_labels.npy")
    BM25_train = numpy.load("WikiQA/train_BM25scores.npy")

    # Load development set
    qids_dev = numpy.load("WikiQA/dev_qids.npy")
    q_dev = numpy.load("WikiQA/dev_questions.npy")
    a_dev = numpy.load("WikiQA/dev_answers.npy")
    y_dev = numpy.load("WikiQA/dev_labels.npy")
    BM25_dev = numpy.load("WikiQA/dev_BM25scores.npy")

    print("q_train shape: ", q_train.shape)
    print("a_train shape: ", a_train.shape)
    print("BM25_train shape: ", BM25_train.shape)
    max_ques_len = q_train.shape[1]
    max_ans_len = a_train.shape[1]
    print("maximum question len: ", max_ques_len)
    print("maximum answer len: ", max_ans_len)

    # Call Model
    global model
    model = SMCNN(max_ques_len, max_ans_len)

    global Q_DEV, A_DEV, Y_DEV, QIDS_DEV, BM_DEV, Q_TRAIN, A_TRAIN, BM_TRAIN, Y_TRAIN, QIDS_TRAIN
    Q_DEV = q_dev
    A_DEV = a_dev
    Y_DEV = y_dev
    BM_DEV = BM25_dev
    QIDS_DEV = qids_dev
    Q_TRAIN = q_train
    A_TRAIN = a_train
    BM_TRAIN = BM25_train
    Y_TRAIN = y_train
    QIDS_TRAIN = qids_train

    map_dev_callback = MAP_dev()
    mrr_dev_callback = MRR_dev()
    map_train_callback = MAP_train()
    mrr_train_callback = MRR_train()
    save_model = save_Model_state()
    history = model.fit([q_train, a_train, BM25_train],
                        y_train, batch_size=200, nb_epoch=3,
                        validation_data=([q_dev, a_dev, BM25_dev], y_dev),
                        callbacks=[map_dev_callback, mrr_dev_callback, map_train_callback, mrr_train_callback, save_model])
    # list all data in history
    print(history.history.keys())

def constructSentencefromIDs(ids, vocab):
    words = []
    for id in ids:
        if id != 0:
            words.append(wordFromID(id, vocab))
    sentence = " ".join(words)
    return sentence


def wordFromID(word_id, vocab):
    word = ""
    for key, value in vocab.items():
        try:
            if value == word_id:
                word = key
                return word
        except:
            print("Error...word not in Vocabulary!!!")


def testTrecQA(best_epoch):
     # Load test set
    qids_test = numpy.load("TrecQA/test_qids.npy")
    q_test = numpy.load("TrecQA/test_questions.npy")
    a_test = numpy.load("TrecQA/test_answers.npy")
    y_test = numpy.load("TrecQA/test_labels.npy")
    BM25_test = numpy.load("TrecQA/test_BM25scores.npy")
    print("")

    # load Model that achieved the best score during training on the development set
    print("Loading best Model...")
    json_file = open("Model_files/model_" + best_epoch + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Model_files/model_" + best_epoch + ".h5")
    global model
    model = loaded_model
    print("Loaded keras Model")

    print("Compiling Model...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', fmeasure])

    print("Evaluating Model on test set...")
    scores = model.evaluate([q_test, a_test, BM25_test], y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    eval_file = open("experiments/evaluation_test.txt", "a")
    map_score, mrr_score = MAP_MRR_score(qids_test, q_test, a_test, y_test, BM25_test)
    eval_file.write(str(model.metrics_names[0]) + "\t"   +  str(scores[0] * 100) + "\n")
    eval_file.write(str(model.metrics_names[1]) + "\t"   +  str(scores[1] * 100)+ "\n")
    eval_file.write(str(model.metrics_names[2]) + "\t"   +  str(scores[2] * 100)+ "\n")
    eval_file.write("MAP" + "\t" + str(map_score * 100)  +  "\n")
    eval_file.write("MRR" + "\t" + str(mrr_score * 100)  +  "\n")
    print("MAP score: ", map_score)
    print("MRR score:", mrr_score)


def testWikiQA(best_epoch):
    qids_test = numpy.load("WikiQA/test_qids.npy")
    q_test = numpy.load("WikiQA/test_questions.npy")
    a_test = numpy.load("WikiQA/test_answers.npy")
    y_test = numpy.load("WikiQA/test_labels.npy")
    BM25_test = numpy.load("WikiQA/test_BM25scores.npy")
    print("")

    # load Model that achieved the best score during training on the development set
    print("Loading best Model...")
    json_file = open("Model_files/model_" + best_epoch + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Model_files/model_" + best_epoch + ".h5")
    global model
    model = loaded_model
    print("Loaded keras Model")

    print("Compiling Model...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', fmeasure])

    print("Evaluating Model on test set...")
    scores = model.evaluate([q_test, a_test, BM25_test], y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
    eval_file = open("experiments/evaluation_test.txt", "a")
    map_score, mrr_score = MAP_MRR_score(qids_test, q_test, a_test, y_test, BM25_test)
    eval_file.write(str(model.metrics_names[0]) + "\t" + str(scores[0] * 100) + "\n")
    eval_file.write(str(model.metrics_names[1]) + "\t" + str(scores[1] * 100) + "\n")
    eval_file.write(str(model.metrics_names[2]) + "\t" + str(scores[2] * 100) + "\n")
    eval_file.write("MAP" + "\t" + str(map_score * 100) + "\n")
    eval_file.write("MRR" + "\t" + str(mrr_score * 100) + "\n")
    print("MAP score: ", map_score)
    print("MRR score:", mrr_score)


def createPredFile(qids_test, q_test, a_test, y_test, BM25_test, old_questions_test, old_answers_test, starts_test, ends_test, dids_test, epoch):

    # Make predictions for the given dataset
    print('q_test shape', q_test.shape)
    print('a_test shape', a_test.shape)
    x_val, y_true = [q_test, a_test, BM25_test], y_test
    y_pred = model.predict(x_val)

    f = open("experiments/predictions.txt", "a")
    f2 = open('vocab.json', 'r')
    vocab = json.load(f2)
    qid2cand = defaultdict(list)
    dict2key = defaultdict(list)
    for qid, question, answer, label, pred, old_q, old_a, old_os, old_oe in zip(qids_test, q_test, a_test, y_true, y_pred, old_questions_test, old_answers_test, starts_test, ends_test):
        qid2cand[qid].append((question, answer, pred, old_q, old_a, old_os, old_oe))
    for qid, question, answer, label, pred, old_q, old_a, old_os, old_oe, doc_id in zip(qids_test, q_test, a_test, y_true, y_pred, old_questions_test, old_answers_test, starts_test, ends_test, dids_test):
        dict2key[qid, old_q].append((answer, pred, old_q, old_a, old_os, old_oe, doc_id))


    print("Creating json file for BioASQ evaluation!!!")
    createJsonFile(dict2key, vocab, epoch)

def createJsonFile(dict, vocab, epoch):

    f = open('experiments/predictions_' + str(epoch) + '.json', 'w')

    data = {'questions': []}
    for keys, candidates in dict.items():
        basic_info = {'body': keys[1], 'id': keys[0], 'snippets': []}
        counter = 0
        for answer, pred, old_q, old_a, os, oe, doc_id in sorted(candidates, key= itemgetter(1), reverse=True):
            if counter < 10:
                snips = {'document': "http://www.ncbi.nlm.nih.gov/pubmed/" + doc_id,
                         'text': old_a,
                         'offsetInBeginSection': int(os),
                         'offsetInEndSection': int(oe),
                         'beginSection': "abstract",
                         'endSection': "abstract"}
                counter += 1
                basic_info['snippets'].append(snips)
        data['questions'].append(basic_info)
    json.dump(data, f, indent=4)


def testBioASQ(best_epoch):
    # Load test set
    qids_test = numpy.load("BioASQ_dataset/test_qids.npy")
    q_test = numpy.load("BioASQ_dataset/test_questions.npy")
    a_test = numpy.load("BioASQ_dataset/test_answers.npy")
    y_test = numpy.load("BioASQ_dataset/test_labels.npy")
    BM25_test = numpy.load("BioASQ_dataset/test_BM25scores.npy")
    old_questions_test = numpy.load("BioASQ_dataset/test_old_questions.npy")
    old_answers_test = numpy.load("BioASQ_dataset/test_old_answers.npy")
    starts_test = numpy.load("BioASQ_dataset/test_starts.npy")
    ends_test = numpy.load("BioASQ_dataset/test_ends.npy")
    dids_test = numpy.load("BioASQ_dataset/test_doc_ids.npy")

    # load Model that achieved the best score during training on the development set
    print("Loading best Model...")
    json_file = open("Model_files/model_" + best_epoch + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Model_files/model_" + best_epoch + ".h5")
    global model
    model = loaded_model
    print("Loaded keras Model")

    print("Compiling Model...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', fmeasure])

    print("Evaluating Model on test set...")

    createPredFile(qids_test, q_test, a_test, y_test, BM25_test, old_questions_test, old_answers_test, starts_test, ends_test, dids_test, "X")


if __name__ == '__main__':
    start_time = time.time()

    train_on_BioASQ()
    testBioASQ('1')

    #train_on_TrecQA()
    #testTrecQA('1')

    # train_on_WikiQA()
    # testWikiQA('1')



    print("--- %s seconds ---" % (time.time() - start_time))




