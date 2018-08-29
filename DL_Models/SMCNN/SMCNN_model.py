from keras.layers.convolutional import Convolution1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.core import Dropout
from keras.models import Model, Sequential
from keras.layers import Dense, Input, merge, LSTM
from keras import initializers
from keras.engine.topology import Layer
import keras.backend as K
import numpy as np
import gensim
import keras
import json


class SimLayer(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		self.init = initializers.get('glorot_uniform')
		super(SimLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[0][1]
		self.W = self.init((input_dim, input_dim))
		self.trainable_weights = [self.W]
		super(SimLayer, self).build(input_shape)

	def call(self, x, mask=None):
		return K.dot(x[0], K.dot(self.W, x[1].T))

	def get_output_shape_for(self, input_shape):
		return (input_shape[0][0], input_shape[0][0])


def load_embeddings(path, vocab):
	word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
	# Retrieve dimension space of embedding vectors
	dim = word2vec['common'].shape[0]
	# Initialize (with zeros) embedding matrix for all vocabulary
	embedding = np.zeros((len(vocab), dim))
	rand_vec = np.random.uniform(-0.25, 0.25, dim)
	rand_count = 0
	# Fill embedding matrix with known embeddings
	for key, value in vocab.items():
		try:
			embedding[value] = word2vec[key]
		except:
			rand_vec = np.random.uniform(-0.25, 0.25, dim)
			embedding[value] = rand_vec
			rand_count += 1

	print("No of random vecs used are %d"%rand_count)
	return embedding, dim


def SMCNN(max_ques_len, max_ans_len):
	f = open('vocab.json', 'r')
	vocab = json.load(f)
	embedding, dim = load_embeddings('./Embeddings/embeddings.bin', vocab)

	inp_q = Input(shape=(max_ques_len,))
	embedding_q = Embedding(len(vocab), dim, input_length=max_ques_len, weights=[embedding], trainable=False)(inp_q)
	conv_q = Convolution1D(100, 5, border_mode='same', activation='relu')(embedding_q)
	conv_q = Dropout(0.25)(conv_q)
	pool_q = GlobalMaxPooling1D()(conv_q)

	inp_a = Input(shape=(max_ans_len,))
	embedding_a = Embedding(len(vocab), dim, input_length=max_ans_len, weights=[embedding], trainable=False)(inp_a)
	conv_a = Convolution1D(100, 5, border_mode='same', activation='relu')(embedding_a)
	conv_a = Dropout(0.25)(conv_a)
	pool_a = GlobalMaxPooling1D()(conv_a)

	sim = merge([Dense(100, bias=False)(pool_q), pool_a], mode='dot')
	BM25_scores = Input(shape=(1,))

	model_sim = merge([pool_q, pool_a, sim, BM25_scores], mode='concat')

	model_final = Dropout(0.5)(model_sim)
	model_final = Dense(202)(model_final)
	model_final = Dropout(0.5)(model_final)
	model_final = Dense(1, activation='sigmoid')(model_final)

	model = Model(input=[inp_q, inp_a, BM25_scores], output=[model_final])

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print("model summary: ", model.summary())
	return model



