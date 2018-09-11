import tensorflow as tf
import sys, json
import numpy as np
from operator import itemgetter

from preprocess import Word2Vec, BioASQ
from ABCNN import ABCNN
from utils import build_path
from sklearn.externals import joblib



def createJsonFile(dict):
    f = open('experiments/final_predictions.json', 'w')
    data = {'questions': []}
    for keys, candidates in dict.items():
        basic_info = {'body': keys[2], 'id': keys[1], 'snippets': []}
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


def constructSentencefromIDs(ids, vocab):
    words = []
    for id in ids:
        if id != 0:
            words.append(wordFromID(id, vocab))
    sentence = " ".join(words)
    return sentence


def wordFromID(word_id, vocab):
    for key, value in vocab.items():
        try:
            if value == word_id:
                word = key
                return word
        except:
            print("Error...word not in Vocabulary!!!")


def test(w, l2_reg, epoch, max_len, model_type, num_layers, data_type, classifier, word2vec, num_classes=2):

    if data_type == 'BioASQ':
        test_data = BioASQ(word2vec=word2vec, max_len=max_len)
    else:
        print("Wrong dataset...")

    test_data.open_file(mode="test")

    model = ABCNN(s=max_len, w=w, l2_reg=l2_reg, model_type=model_type,
                  num_features=test_data.num_features, num_classes=num_classes, num_layers=num_layers)

    model_path = build_path("./models/", data_type, model_type, num_layers)

    MAPs, MRRs = [], []

    print("=" * 50)
    print("test data size:", test_data.data_size)

    test_data.reset_index()

    with tf.Session() as sess:
        max_epoch = 2    # Enter the best epoch during the evaluation of the validation set
        saver = tf.train.Saver()
        saver.restore(sess, model_path + "-" + str(max_epoch))
        print(model_path + "-" + str(max_epoch), "restored.")

        if classifier == "LR" or classifier == "SVM":
            clf_path = build_path("./models/", data_type, model_type, num_layers,
                                  "-" + str(max_epoch) + "-" + classifier + ".pkl")
            clf = joblib.load(clf_path)
            print(clf_path, "restored.")

        QA_pairs = {}
        Bio_pairs = {}

        s1s, s2s, labels, features = test_data.next_batch(batch_size=test_data.data_size)
        qids, old_qs, old_as, starts, ends, dids = test_data.getMoreInfo()

        for i in range(test_data.data_size):
            pred, clf_input = sess.run([model.prediction, model.output_features],
                                       feed_dict={model.x1: np.expand_dims(s1s[i], axis=0),
                                                  model.x2: np.expand_dims(s2s[i], axis=0),
                                                  model.y: np.expand_dims(labels[i], axis=0),
                                                  model.features: np.expand_dims(features[i], axis=0)})

            if classifier == "LR":
                clf_pred = clf.predict_proba(clf_input)[:, 1]
                pred = clf_pred
            elif classifier == "SVM":
                clf_pred = clf.decision_function(clf_input)
                pred = clf_pred

            s1 = " ".join(test_data.s1s[i])
            s2 = " ".join(test_data.s2s[i])

            if s1 in QA_pairs:
                QA_pairs[s1].append((s2, labels[i], np.asscalar(pred)))
                Bio_pairs[(s1, qids[i], old_qs[i])].append((s2, np.asscalar(pred), old_qs[i], old_as[i], starts[i], ends[i], dids[i]))
            else:
                QA_pairs[s1] = [(s2, labels[i], np.asscalar(pred))]
                Bio_pairs[(s1, qids[i], old_qs[i])] = [(s2, np.asscalar(pred), old_qs[i], old_as[i], starts[i], ends[i], dids[i])]

        createJsonFile(Bio_pairs)

        # Calculate MAP and MRR
        MAP, MRR = 0, 0
        for s1 in QA_pairs.keys():
            p, AP = 0, 0
            MRR_check = False

            QA_pairs[s1] = sorted(QA_pairs[s1], key=lambda x: x[-1], reverse=True)

            for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
                if label == 1:
                    if not MRR_check:
                        MRR += 1 / (idx + 1)
                        MRR_check = True

                    p += 1
                    AP += p / (idx + 1)

            if p != 0:
                AP /= p
            else:
                AP = 0
            MAP += AP

        num_questions = len(QA_pairs.keys())
        MAP /= num_questions
        MRR /= num_questions

        MAPs.append(MAP)
        MRRs.append(MRR)

if __name__ == "__main__":

    # Parameters
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --max_len: max sentence length
    # --model_type: model type
    # --num_layers: number of convolution layers
    # --data_type: dataset name
    # --classifier: Final layout classifier(model, LR, SVM)

    # default parameters
    params = {
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 3,
        "max_len": 40,
        "model_type": "BCNN",
        "num_layers": 2,
        "data_type": "BioASQ",
        "classifier": "LR",
        "word2vec": Word2Vec()
    }

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    test(w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
         max_len=int(params["max_len"]), model_type=params["model_type"],
         num_layers=int(params["num_layers"]), data_type=params["data_type"],
         classifier=params["classifier"], word2vec=params["word2vec"])
