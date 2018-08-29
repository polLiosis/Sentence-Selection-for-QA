import BM25, Data

# Path for datasets
train_set = "Datasets/BioASQ/train.txt"
test_set = "Datasets/BioASQ/test.txt"

# Load df and idf scores from pickle files
df_scores, idf_scores = Data.load_idf_scores()

# Load Data
train_qids, train_questions, train_documents, train_labels = Data.load_dataset(train_set)
test_qids, test_questions, test_documents, test_labels = Data.load_dataset(test_set)

# Compute average document length for each dataset
train_avgdl = BM25.compute_avgdl(train_documents)
test_avgdl = BM25.compute_avgdl(test_documents)

# Compute mean and deviation for Z-score normalization
max = max(idf_scores.keys(), key=(lambda i: idf_scores[i]))
rare_word_value = idf_scores[max]
mean, deviation = BM25.compute_Zscore_values(train_set, idf_scores, train_avgdl, 1.2, 0.75, rare_word_value)

# Create BM25 scores (Using BioASQ format)
BM25.createBioASQformat(test_set, idf_scores, test_avgdl, True, mean, deviation, rare_word_value)