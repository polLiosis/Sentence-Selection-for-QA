import json, pickle, BM25
from collections import defaultdict
import numpy as np

def load_data(train_set, dev_set, test_set):
	total_qids, total_questions, total_answers, total_labels = [], [], [], []
	train_qids, train_questions, train_answers, train_labels = [], [], [], []
	dev_qids, dev_questions, dev_answers, dev_labels = [], [], [], []
	test_qids, test_questions, test_answers, test_labels = [], [], [], []

	with open(train_set, "r", encoding="utf-8") as train_f:
		for line in train_f:
			items = line[:-1].split("\t")
			qid = items[0]
			question = items[1].lower().split()
			answer = items[2].lower().split()
			label = int(items[3])

			total_qids.append(qid)
			total_questions.append(question)
			total_answers.append(answer)
			total_labels.append(label)
			train_qids.append(qid)
			train_questions.append(question)
			train_answers.append(answer)
			train_labels.append(label)

	with open(dev_set, "r", encoding="utf-8") as dev_f:
		for line in dev_f:
			items = line[:-1].split("\t")
			qid = items[0]
			question = items[1].lower().split()
			answer = items[2].lower().split()
			label = int(items[3])

			total_qids.append(qid)
			total_questions.append(question)
			total_answers.append(answer)
			total_labels.append(label)
			dev_qids.append(qid)
			dev_questions.append(question)
			dev_answers.append(answer)
			dev_labels.append(label)

	with open(test_set, "r", encoding="utf-8") as test_f:
		for line in test_f:
			items = line[:-1].split("\t")
			qid = items[0]
			question = items[1].lower().split()
			answer = items[2].lower().split()
			label = int(items[3])

			total_qids.append(qid)
			total_questions.append(question)
			total_answers.append(answer)
			total_labels.append(label)
			test_qids.append(qid)
			test_questions.append(question)
			test_answers.append(answer)
			test_labels.append(label)

	return total_qids, total_questions, total_answers, total_labels, train_qids, train_questions, train_answers, train_labels, dev_qids, dev_questions, dev_answers, dev_labels, test_qids, test_questions, test_answers, test_labels


def gen_vocab(data):
	vocab = defaultdict(int)
	vocab_idx = 1
	for component in data:
		for text in component:
			for token in text:
				if token not in vocab:
					vocab[token] = vocab_idx
					vocab_idx += 1
	vocab['UNK'] = len(vocab)
	f = open('vocab.json', 'w')
	json.dump(vocab, f)
	return vocab


def get_maxlen(data):
	return max(map(lambda x:len(x), data))


def gen_seq(data, vocab, max_len):
	X = []
	for text in data:
		temp = [0] * max_len
		temp[:len(text)] = map(lambda x:vocab.get(x, vocab['UNK']), text)
		X.append(temp)
	X = np.array(X)
	return X


if __name__ == '__main__':
	train_set = "WikiQA/train.txt"
	dev_set = "WikiQA/dev.txt"
	test_set = "WikiQA/test.txt"

	# We retrieve the data from the text files
	qids, questions, answers, labels, train_qids, train_questions, train_answers, train_labels, dev_qids, dev_questions, dev_answers, dev_labels, test_qids, test_questions, test_answers, test_labels = load_data(train_set, dev_set, test_set)
	vocab = gen_vocab([questions, answers])
	max_len_ques = get_maxlen(questions)
	max_len_ans = get_maxlen(answers)


	BM25scores_train = []
	BM25scores_dev = []
	BM25scores_test = []
	with open('BM25_files/idf.pkl', 'rb') as f:
		idfs = pickle.load(f)
	with open('BM25_files/df.pkl', 'rb') as f:
		dfs = pickle.load(f)

	max = max(idfs.keys(), key=(lambda i: idfs[i]))
	rare_word_value = idfs[max]

	# BM scores for train set
	avgdl = BM25.compute_avgdl(train_answers)
	mean, deviation = BM25.compute_Zscore_values(train_set, idfs, avgdl, 1.2, 0.75, rare_word_value)

	file = open("BM25/train_BM25scores.txt", "a")
	file.write(str(avgdl) + "\n")
	for q, a in zip(train_questions, train_answers):
		query = q
		candidate_answer = a
		BM25_score = BM25.similarity_score(query, candidate_answer, 1.2, 0.75, idfs, avgdl, True, mean, deviation,rare_word_value)
		file.write(str(query) + "\t" + str(candidate_answer) + "\t" + str(BM25_score) + " \n")
		BM25scores_train.append(BM25_score)
	file.close()

	# BM scores for development set
	avgdl = BM25.compute_avgdl(dev_answers)
	file = open("BM25/dev_BM25scores.txt", "a")
	file.write(str(avgdl) + "\n")
	for q, a in zip(dev_questions, dev_answers):
		query = q
		candidate_answer = a
		BM25_score = BM25.similarity_score(query, candidate_answer, 1.2, 0.75, idfs, avgdl, True, mean, deviation, rare_word_value)
		file.write(str(query) + "\t" + str(candidate_answer) + "\t" + str(BM25_score) + " \n")
		BM25scores_dev.append(BM25_score)
	file.close()

	# BM scores for test set
	avgdl = BM25.compute_avgdl(test_answers)
	file = open("BM25/test_BM25scores.txt", "a")
	file.write(str(avgdl) + "\n")
	for q, a in zip(test_questions, test_answers):
		query = q
		candidate_answer = a
		BM25_score = BM25.similarity_score(query, candidate_answer, 1.2, 0.75, idfs, avgdl, True, mean, deviation, rare_word_value)
		file.write(str(query) + "\t" + str(candidate_answer) + "\t" + str(BM25_score) + " \n")
		BM25scores_test.append(BM25_score)
	file.close()

	gen_train_questions  =  gen_seq(train_questions, vocab, max_len_ques)
	gen_dev_questions    =  gen_seq(dev_questions, vocab, max_len_ques)
	gen_test_questions   =  gen_seq(test_questions, vocab, max_len_ques)
	gen_train_answers    =  gen_seq(train_answers, vocab, max_len_ans)
	gen_dev_answers      =  gen_seq(dev_answers, vocab, max_len_ans)
	gen_test_answers     =  gen_seq(test_answers, vocab, max_len_ans)


	np.save("WikiQA/train_qids.npy", train_qids)
	np.save("WikiQA/train_questions.npy", gen_train_questions)
	np.save("WikiQA/train_answers.npy", gen_train_answers)
	np.save("WikiQA/train_labels.npy", train_labels)
	np.save("WikiQA/train_BM25scores.npy", np.array(BM25scores_train).astype(float))

	np.save("WikiQA/dev_qids.npy", dev_qids)
	np.save("WikiQA/dev_questions.npy", gen_dev_questions)
	np.save("WikiQA/dev_answers.npy", gen_dev_answers)
	np.save("WikiQA/dev_labels.npy", dev_labels)
	np.save("WikiQA/dev_BM25scores.npy", np.array(BM25scores_dev).astype(float))

	np.save("WikiQA/test_qids.npy", test_qids)
	np.save("WikiQA/test_questions.npy", gen_test_questions)
	np.save("WikiQA/test_answers.npy", gen_test_answers)
	np.save("WikiQA/test_labels.npy", test_labels)
	np.save("WikiQA/test_BM25scores.npy", np.array(BM25scores_test).astype(float))