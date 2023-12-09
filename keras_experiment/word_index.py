from keras_experiment.typings import TokenLabel, ValueKey, KeyValue, kUNK, kPAD


class Word2Ind:
    def load_words(self, tokens_labels: list[TokenLabel]) -> tuple[list[str], ValueKey, KeyValue]:

        word_set: set[str] = set()

        for q_text, label in tokens_labels:
            for w in q_text:
                word_set.add(w)

        words = sorted(word_set)
        words = [kPAD, kUNK] + words

        word2ind: ValueKey = {}
        ind2word: KeyValue = {}
        for w in words:
            index = len(word2ind)
            word2ind[w] = index
            ind2word[index] = w
        return words, word2ind, ind2word

    def class_labels(self, data) -> tuple[ValueKey, KeyValue]:
        class_to_i: ValueKey = {}
        i_to_class: KeyValue = {}

        unique_classes: set[str] = set()

        for question, ans in data:

            if ans in unique_classes:
                continue
            i = len(unique_classes)
            class_to_i[ans] = i
            i_to_class[i] = ans
            unique_classes.add(ans)
        return class_to_i, i_to_class