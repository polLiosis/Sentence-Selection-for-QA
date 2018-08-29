import json, pickle, BM25
from collections import defaultdict
import numpy as np

# Load BioASQ dataset (train, dev and test set)
def load_data(train_set, dev_set, test_set):
	total_qids, total_questions, total_answers, total_labels = [], [], [], []
	train_qids, train_questions, train_answers, train_labels = [], [], [], []
	dev_qids, dev_questions, dev_answers, dev_old_questions, dev_old_answers, dev_starts, dev_ends, dev_labels, dev_psnips, dev_rsnips, dev_dids = [], [], [], [], [], [], [], [], [], [], []
	test_qids, test_questions, test_answers, test_old_questions, test_old_answers, test_starts, test_ends, test_labels, test_psnips, test_rsnips, test_dids = [], [], [], [], [], [], [], [], [], [], []

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

			psnip = 0.0
			rsnip = 0.0
			label = 0
			old_question = items[3]
			old_answer = items[4]
			start = items[5]
			end = items[6]
			did = items[7]

			dev_old_questions.append(old_question)
			dev_old_answers.append(old_answer)
			dev_starts.append(start)
			dev_ends.append(end)
			dev_dids.append(did)
			total_qids.append(qid)
			total_questions.append(question)
			total_answers.append(answer)
			total_labels.append(label)
			dev_qids.append(qid)
			dev_questions.append(question)
			dev_answers.append(answer)
			dev_labels.append(label)
			dev_psnips.append(psnip)
			dev_rsnips.append(rsnip)

	with open(test_set, "r", encoding="utf-8") as test_f:
		for line in test_f:
			items = line[:-1].split("\t")
			qid = items[0]
			question = items[1].lower().split()
			answer = items[2].lower().split()

			psnip = 0.0
			rsnip = 0.0
			label = 0
			old_question = items[3]
			old_answer = items[4]
			start = items[5]
			end = items[6]
			did = items[7]

			test_old_questions.append(old_question)
			test_old_answers.append(old_answer)
			test_starts.append(start)
			test_ends.append(end)
			test_dids.append(did)
			total_qids.append(qid)
			total_questions.append(question)
			total_answers.append(answer)
			total_labels.append(label)
			test_qids.append(qid)
			test_questions.append(question)
			test_answers.append(answer)
			test_labels.append(label)
			test_psnips.append(psnip)
			test_rsnips.append(rsnip)

	return total_qids, total_questions, total_answers, total_labels, train_qids, train_questions, train_answers, train_labels, dev_qids, dev_questions, dev_answers, dev_old_questions, dev_old_answers, dev_starts, dev_ends, dev_labels, dev_psnips, dev_rsnips, dev_dids, test_qids, test_questions, test_answers, test_old_questions, test_old_answers, test_starts, test_ends, test_labels, test_psnips, test_rsnips, test_dids

# Generate Vocabulary from given dataset
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

# Get maximum length
def get_maxlen(data):
	return max(map(lambda x:len(x), data))

# Generate a sequence if ids (based on the words in our vocabulary)
def gen_seq(data, vocab, max_len, type):
	X = []
	for text in data:
		temp = [0] * max_len
		temp[:len(text)] = map(lambda x:vocab.get(x, vocab['UNK']), text)
		X.append(temp)
	X = np.array(X)
	return X




if __name__ == '__main__':
	train_set = "BioASQ_dataset/train.txt"
	dev_set = "BioASQ_dataset/dev.txt"
	test_set = "BioASQ_dataset/test.txt"

    # Retrieve the data from the text files
	qids, questions, answers, labels, train_qids, train_questions, train_answers, train_labels, dev_qids, dev_questions, dev_answers, dev_old_questions, dev_old_answers, dev_starts, dev_ends, dev_labels, dev_psnips, dev_rsnips, dev_doc_ids, test_qids, test_questions, test_answers, test_old_questions, test_old_answers, test_starts, test_ends, test_labels, test_psnips, test_rsnips, test_doc_ids  = load_data(train_set, dev_set, test_set)
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
		BM25_score = BM25.similarity_score(query, candidate_answer, 1.2, 0.75 , idfs, avgdl, True, mean, deviation, rare_word_value)
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

    # Generate sequences
	gen_train_questions  =  gen_seq(train_questions, vocab, max_len_ques, 'question')
	gen_dev_questions    =  gen_seq(dev_questions, vocab, max_len_ques, 'question')
	gen_test_questions   =  gen_seq(test_questions, vocab, max_len_ques, 'question')
	gen_train_answers    =  gen_seq(train_answers, vocab, max_len_ans, 'answer')
	gen_dev_answers      =  gen_seq(dev_answers, vocab, max_len_ans, 'answer')
	gen_test_answers     =  gen_seq(test_answers, vocab, max_len_ans, 'answer')

	# Save data
	np.save("BioASQ_dataset/train_qids.npy", train_qids)
	np.save("BioASQ_dataset/train_questions.npy", gen_train_questions)
	np.save("BioASQ_dataset/train_answers.npy", gen_train_answers)
	np.save("BioASQ_dataset/train_labels.npy", train_labels)
	np.save("BioASQ_dataset/train_BM25scores.npy", np.array(BM25scores_train).astype(float))

	np.save("BioASQ_dataset/dev_qids.npy", dev_qids)
	np.save("BioASQ_dataset/dev_questions.npy", gen_dev_questions)
	np.save("BioASQ_dataset/dev_answers.npy", gen_dev_answers)
	np.save("BioASQ_dataset/dev_labels.npy", dev_labels)
	np.save("BioASQ_dataset/dev_BM25scores.npy", np.array(BM25scores_dev).astype(float))
	np.save("BioASQ_dataset/dev_psnips.npy", dev_psnips)
	np.save("BioASQ_dataset/dev_rsnips.npy", dev_rsnips)
	np.save("BioASQ_dataset/dev_old_questions.npy", dev_old_questions)
	np.save("BioASQ_dataset/dev_old_answers.npy", dev_old_answers)
	np.save("BioASQ_dataset/dev_starts.npy", dev_starts)
	np.save("BioASQ_dataset/dev_ends.npy", dev_ends)
	np.save("BioASQ_dataset/dev_doc_ids.npy", dev_doc_ids)

	np.save("BioASQ_dataset/test_qids.npy", test_qids)
	np.save("BioASQ_dataset/test_questions.npy", gen_test_questions)
	np.save("BioASQ_dataset/test_answers.npy", gen_test_answers)
	np.save("BioASQ_dataset/test_labels.npy", test_labels)
	np.save("BioASQ_dataset/test_BM25scores.npy", np.array(BM25scores_test).astype(float))
	np.save("BioASQ_dataset/test_psnips.npy", test_psnips)
	np.save("BioASQ_dataset/test_rsnips.npy", test_rsnips)
	np.save("BioASQ_dataset/test_old_questions.npy", test_old_questions)
	np.save("BioASQ_dataset/test_old_answers.npy", test_old_answers)
	np.save("BioASQ_dataset/test_starts.npy", test_starts)
	np.save("BioASQ_dataset/test_ends.npy", test_ends)
	np.save("BioASQ_dataset/test_doc_ids.npy", test_doc_ids)