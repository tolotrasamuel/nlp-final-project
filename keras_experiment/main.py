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


voc, word2ind, ind2word = Word2Ind().load_words(text_tokens)
class2ind, ind2class = Word2Ind().class_labels(text_tokens)

test_dataset = QuestionDataset(text_tokens, word2ind, class2ind)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Pad sequences to ensure consistent length
# Define the model
embedding_dim = 16

model = Sequential()
model.add(
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10, batch_size=2)

# Make predictions
new_texts = ["A new positive statement.", "This seems negative."]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)

predictions = model.predict(new_padded_sequences)

for text, prediction in zip(new_texts, predictions):
    print(f'Text: {text} | Predicted Probability: {prediction[0]}')
