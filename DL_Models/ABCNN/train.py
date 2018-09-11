import tensorflow as tf
import numpy as np
import sys

from preprocess import Word2Vec, BioASQ
from ABCNN import ABCNN
from utils import build_path
from sklearn import linear_model, svm
from sklearn.externals import joblib

N_CORRECT = 0
N_ITEMS_SEEN = 0

def reset_running_variables():
    global N_CORRECT, N_ITEMS_SEEN
    N_CORRECT = 0
    N_ITEMS_SEEN = 0

def update_running_variables(labs, preds):
    global N_CORRECT, N_ITEMS_SEEN
    N_CORRECT += (labs == preds).sum()
    N_ITEMS_SEEN += labs.size

def calculate_accuracy():
    global N_CORRECT, N_ITEMS_SEEN
    return float(N_CORRECT) / N_ITEMS_SEEN

def train(lr, w, l2_reg, epoch, batch_size, model_type, num_layers, data_type, word2vec, num_classes=2):

    if data_type == 'BioASQ':
        train_data = BioASQ(word2vec=word2vec)
    else:
        print("Wrong dataset...")

    # We open the train text file in train mode
    train_data.open_file(mode="train")

    print("=" * 50)
    # Get total lines in the train text file (QA pairs)
    print("training data size:", train_data.data_size)
    print("training max len:", train_data.max_len)
    print("=" * 50)

    model = ABCNN(s=train_data.max_len, w=w, l2_reg=l2_reg, model_type=model_type,
                  num_features=train_data.num_features, num_classes=num_classes, num_layers=num_layers)

    # We use Adagrad optimizer for our train process
    optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.cost)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # The Saver class adds ops to save and restore variables to and from checkpoints.
    # Checkpoints are binary files in a proprietary format which map variable names to tensor values.
    # The best way to examine the contents of a checkpoint is to load it using a Saver.
    saver = tf.train.Saver(max_to_keep=100)

    # A class for running TensorFlow operations.
    # A Session object encapsulates the environment in which Operation objects are executed,
    # and Tensor objects are evaluated.
    with tf.Session() as sess:

        sess.run(init)    # initialize variables

        print("=" * 50)

        # We start the loop for training, which is based on the number of epochs we entered as a parameter
        for e in range(1, epoch + 1):
            print("[Epoch " + str(e) + "]")
            train_data.reset_index()
            i = 0
            LR = linear_model.LogisticRegression()
            SVM = svm.LinearSVC()
            clf_features = []

            # While we have data to train our model
            while train_data.is_available():
                i += 1

                # We retrieved the next training batch
                # batch_x1 : Question sentence
                # batch_x2 : Candidate answer sentence
                # batch-y : label (0 or 1)
                # batch_features : batch features
                batch_x1, batch_x2, batch_y, batch_features = train_data.next_batch(batch_size=batch_size)
                merged, _, c, features = sess.run([model.merged, optimizer, model.cost, model.output_features],
                                                  feed_dict={model.x1: batch_x1,
                                                             model.x2: batch_x2,
                                                             model.y: batch_y,
                                                             model.features: batch_features})

                clf_features.append(features)

                if i % 100 == 0:
                    print("[batch " + str(i) + "] cost:", c)

            # Save info for the specific epoch
            save_path = saver.save(sess, build_path("./models/", data_type, model_type, num_layers), global_step=e)
            print("model saved as", save_path)

            clf_features = np.concatenate(clf_features)
            LR.fit(clf_features, train_data.labels)
            SVM.fit(clf_features, train_data.labels)

            LR_path = build_path("./models/", data_type, model_type, num_layers, "-" + str(e) + "-LR.pkl")
            SVM_path = build_path("./models/", data_type, model_type, num_layers, "-" + str(e) + "-SVM.pkl")
            joblib.dump(LR, LR_path)
            joblib.dump(SVM, SVM_path)

            print("LR saved as", LR_path)
            print("SVM saved as", SVM_path)

        # When the loop ends, our model was trained with success!!!
        print("training finished!")
        print("=" * 50)


if __name__ == "__main__":

    # Paramters
    # --lr: learning rate
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --batch_size: batch size
    # --model_type: model type
    # --num_layers: number of convolution layers
    # --data_type: dataset with which we want to train our model

    # default parameters
    params = {
        "lr": 0.08,
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 3,
        "batch_size": 200,
        "model_type": "BCNN",
        "num_layers": 2,
        "data_type": "BioASQ",
        "word2vec": Word2Vec()
    }

    print("=" * 50)
    print("Parameters:")
    for k in sorted(params.keys()):
        print(k, ":", params[k])

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    # Call train function to train our model
    train(lr=float(params["lr"]), w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
          batch_size=int(params["batch_size"]), model_type=params["model_type"], num_layers=int(params["num_layers"]),
          data_type=params["data_type"], word2vec=params["word2vec"])
