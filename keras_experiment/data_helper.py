from typing import Optional

import nltk

from keras_experiment.file_helper import FileHelper
from keras_experiment.typings import TokenLabel


prediction_task = "category"
class DataHelper :

    @staticmethod
    def load_and_tokenize_data(filename: str, limit: Optional[int] = None) -> list[TokenLabel]:
        # texts = [
        #     ("This is a positive example.", "positive"),
        #     ("Negative sentiment detected here.", "negative"),
        #     ("Another positive statement.", "positive"),
        #     ("This is a negative one.", "negative"),
        # ]
        # return texts;

        print("Loading data from %s" % filename)

        tokens_labels: list[TokenLabel] = []
        dataset = FileHelper.read_json_file(filename)
        questions_labels = dataset["questions"]
        all_count = len(questions_labels)
        if limit is not None and limit > 0:
            questions_labels = questions_labels[:limit]

        print(f"Tokenizing {len(questions_labels)} out of {all_count} questions")

        for q in questions_labels:
            q_text = nltk.word_tokenize(q['text'])
            if prediction_task == "category":
                label = q['category']
            else:
                label = q['page']

            if label is None or label == "":
                continue
            tokens_labels.append((q_text, label))

        print("Loaded %d data from %s" % (len(tokens_labels), filename))
        return tokens_labels