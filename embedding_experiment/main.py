import numpy as np

from dropout_layer import DropoutLayer
from embedding_experiment.torch_loss import torch_cross_entropy_loss_and_gradient
from fc_layer_neurone import FCLayerDetailed
from keras_experiment.data_helper import DataHelper
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, relu_prime, relu, softmax, softmax_prime
from losses import mse, mse_prime, crossentropyloss, crossentropyloss_prime, cross_entropy_loss_function
from simple_embedding_layer import SimpleEmbeddingLayer, GlobalAveragePooling1D

import nltk as nltk

from keras_experiment.nlp_dataset import QuestionDataset
from keras_experiment.typings import TokenLabel
from keras_experiment.word_index import Word2Ind

# Sample data
texts = [
    ("This is a positive example.", "positive"),
    ("Negative sentiment detected here.", "negative"),
    ("Another positive statement.", "positive"),
    ("This is a negative one.", "negative"),
]

text_tokens: list[TokenLabel] = []
for text in texts:
    t = nltk.word_tokenize(text[0])
    text_tokens.append((t, text[1]))



data_path = "/Users/samuel/PycharmProjects/nlp_class/ast4/data/qanta.train.json"
text_tokens = DataHelper.load_and_tokenize_data(data_path, 1000)

voc, word2ind, ind2word = Word2Ind().load_words(text_tokens)
class2ind, ind2class = Word2Ind().class_labels(text_tokens)

test_dataset = QuestionDataset(text_tokens, word2ind, class2ind)

emb_dim = 50
n_hidden_units = 50
nn_dropout = 0.5
# network
num_words = len(word2ind)
n_classes = len(class2ind)
net = Network(verbose=False)
net.add(SimpleEmbeddingLayer(num_embeddings=num_words, embedding_dim=emb_dim))
net.add(GlobalAveragePooling1D())
# net.add(FCLayerDetailed(input_size=emb_dim, output_size=n_hidden_units))
net.add(FCLayer(input_size=emb_dim, output_size=n_hidden_units))
net.add(ActivationLayer(relu, relu_prime))
net.add(DropoutLayer(nn_dropout))
# net.add(FCLayerDetailed(input_size=n_hidden_units, output_size=n_classes))
net.add(FCLayer(input_size=n_hidden_units, output_size=n_classes))
# net.add(ActivationLayer(relu, relu_prime))
net.add(ActivationLayer(softmax, softmax_prime))
# net._softmax = softmax
# train
net.use(cross_entropy_loss_function, crossentropyloss_prime)
learning_rate = 0.1
epochs = 100
# net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)
# sample dimension first
x_train = []
y_train = []

for i in range(len(test_dataset)):
    x_train.append(np.array([np.array(test_dataset[i][0])]))
    y_train.append([[test_dataset[i][1]]])
samples = len(x_train)
y_train = np.array(y_train)
# training loop
for i in range(epochs):
    err = 0
    for j in range(samples):
        # forward propagation
        output = net.predict_single(x_train[j])

        # compute loss (for display purpose only)
        # err += net.loss(y_train[j], output)
        loss, gradient = torch_cross_entropy_loss_and_gradient(y_train[j][0], output[0])
        err += loss

        # backward propagation
        # error = net.loss_prime(y_train[j], output)
        error = gradient
        # print(err)
        # if err is nan:
        if np.isnan(err):
            print("error is nan")
        if np.isinf(err):
            print("error is inf")
        for layer in reversed(net.layers):
            if net.verbose:
                print("backward",error.shape, layer)
            #     check if has nan
            if np.isnan(error).any():
                print("error is nan")
            error = layer.backward_propagation(error, learning_rate)

    # calculate average error on all samples
    err /= samples
    print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
# test
# out = net.predict(x_train)
# print(out)
