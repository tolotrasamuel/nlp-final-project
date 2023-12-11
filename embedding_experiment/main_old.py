import pickle
from collections import Counter

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

train_text_tokens: list[TokenLabel] = []
for text in texts:
    t = nltk.word_tokenize(text[0])
    train_text_tokens.append((t, text[1]))



# train_data_path = "/Users/samuel/PycharmProjects/nlp_class/ast4/data/qanta.train.json"
train_data_path = "../data/data_5000.csv"
train_text_tokens = DataHelper.load_and_tokenize_data(train_data_path, 500)
voc, word2ind, ind2word = Word2Ind().load_words(train_text_tokens)
class2ind, ind2class = Word2Ind().class_labels(train_text_tokens)
train_dataset = QuestionDataset(train_text_tokens, word2ind, class2ind)

# count per label
counter = Counter()
for q_text, label in train_text_tokens:
    counter[label] += 1
print(counter)

# dev_data_path = "/Users/samuel/PycharmProjects/nlp_class/ast4/data/qanta.dev.json"
dev_data_path ="../data/data_5000.dev.csv"
dev_text_tokens = DataHelper.load_and_tokenize_data(dev_data_path, 250)
dev_dataset = QuestionDataset(dev_text_tokens, word2ind, class2ind)
# class DataHelper:
# class DataHelper:
# count per label
counter = Counter()
for q_text, label in dev_text_tokens:
    counter[label] += 1
print(counter)

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
# net.add(ActivationLayer(tanh, tanh_prime))
# net.add(FCLayer(input_size=emb_dim, output_size=n_hidden_units))
# net.add(ActivationLayer(tanh, tanh_prime))
# # net.add(ActivationLayer(tanh, tanh_prime))
# # net.add(DropoutLayer(nn_dropout))
# # net.add(FCLayerDetailed(input_size=n_hidden_units, output_size=n_classes))
# net.add(FCLayer(input_size=n_hidden_units, output_size=n_hidden_units))
# net.add(ActivationLayer(tanh, tanh_prime))
# net.add(FCLayer(input_size=n_hidden_units, output_size=n_classes))

# net.add(ActivationLayer(relu, relu_prime))
# net.add(ActivationLayer(softmax, softmax_prime))
# net._softmax = softmax
# train

net.add(FCLayer(input_size=emb_dim, output_size=n_hidden_units))
net.add(ActivationLayer(relu, relu_prime))
net.add(DropoutLayer(nn_dropout))
# net.add(FCLayerDetailed(input_size=n_hidden_units, output_size=n_classes))
net.add(FCLayer(input_size=n_hidden_units, output_size=n_classes))
# net.add(ActivationLayer(relu, relu_prime))
net.add(ActivationLayer(softmax, softmax_prime))
# net._softmax = softmax


net.use(mse, mse_prime)
learning_rate = 0.01
epochs = 100
# net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)
# sample dimension first
x_train = []
y_train = []

for i in range(len(train_dataset)):
    x_train.append(np.array([np.array(train_dataset[i][0])]))
    y_train.append([[train_dataset[i][1]]])
samples = len(x_train)
y_train = np.array(y_train)
# training loop
check_acc = 5


def calculate_acc():
    dataset = dev_dataset
    # dataset = train_dataset
    correct = 0
    accuracy_per_label = Counter()
    test_per_label = Counter()


    for i in range(len(dataset)):
        pred = net.predict_single(np.array([dataset[i][0]]))
        # argmex = softmax(pred[0])
        # print(np.argmax(pred) )
        # print(pred[0])
        test_per_label[ind2class[dataset[i][1]]] += 1
        if np.argmax(pred) == dataset[i][1]:
            correct += 1
            accuracy_per_label[ind2class[dataset[i][1]]] += 1
        print(f"acc: {correct / len(dataset)}", end="\r")
    print("acc: ", correct / len(dataset))
    acc = {}
    for key in accuracy_per_label:
        acc[key] = accuracy_per_label[key] / test_per_label[key]
    print(acc)


def save_model(net, param):
    print("saving model")
    pickle.dump(net, open(param, "wb"))


for i in range(epochs):
    err = 0
    for j in range(samples):
        print("epoch %d/%d   sample %d/%d" % (i + 1, epochs, j + 1, samples), end="\r")
        # forward propagation
        output = net.predict_single(x_train[j])

        # compute loss (for display purpose only)
        # err += net.loss(y_train[j], output)
        # error = net.loss_prime(y_train[j], output)

        # sent = [ind2word[x] for x in x_train[j][0]]
        loss, gradient = torch_cross_entropy_loss_and_gradient(y_train[j][0], output[0])
        err += loss
        error = np.array([gradient])


        # print(err)
        # if err is nan:
        # backward propagation
        if np.isnan(err):
            print("error is nan")
        if np.isinf(err):
            print("error is inf")
        for layer in reversed(net.layers):
            if net.verbose:
                print("backward", error.shape, layer)
            #     check if has nan
            if np.isnan(error).any():
                print("error is nan")
            error = layer.backward_propagation(error, learning_rate)

    # calculate average error on all samples
    err /= samples
    print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
    calculate_acc()
    save_model(net, f"model_{i}.json")

# test
# out = net.predict(x_train)
# print(out)
