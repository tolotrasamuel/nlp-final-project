from keras_experiment.typings import TokenLabel, ValueKey, VectorInt, kUNK


class QuestionDataset:

    def __init__(self, examples: list[TokenLabel], word2ind: ValueKey, class2ind: ValueKey):
        self.questions: list[list[str]] = []
        labels: list[str] = []
        for token_label in examples:
            token, label = token_label
            self.questions.append(token)
            labels.append(label)

        self.vectorized_labels = [0] * len(labels)

        unk_class_index = len(class2ind)

        for i in range(len(self.vectorized_labels)):
            try:
                self.vectorized_labels[i] = class2ind[labels[i]]
            except KeyError:
                self.vectorized_labels[i] = unk_class_index
        self.word2ind = word2ind

    def __getitem__(self, index: int) -> tuple[VectorInt, int]:
        sent_vec = self.vectorize(self.questions[index], self.word2ind)
        return sent_vec, self.vectorized_labels[index]

    def __len__(self):
        return len(self.questions)

    @staticmethod
    def vectorize(token_sent: list[str], word2ind: ValueKey) -> VectorInt:

        vec_output = [0] * len(token_sent)

        for i in range(len(token_sent)):
            word = token_sent[i]
            word_is_in_dict = word in word2ind
            if not word_is_in_dict:

                unk_index = word2ind[kUNK]
                vec_output[i] = unk_index
            else:
                vec_output[i] = word2ind[word]
        return vec_output