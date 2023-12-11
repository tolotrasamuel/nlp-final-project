from collections import Counter
from typing import Optional

import nltk
import numpy as np
import pandas as pd

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
        dataset = pd.read_csv(filename)
        # questions_labels = dataset["questions"]
        questions_labels = dataset
        all_count = len(questions_labels)
        # if limit is not None and limit > 0:
        #     questions_labels = questions_labels[:limit]
        # np.where(questions_labels == None)
        print(f"Tokenizing {len(questions_labels)} out of {all_count} questions")
        overall_counter = Counter()
        # print("")
        labels = {"Literature", "Science", "History"}
        labels = None
        counter = Counter()
        for i in range(len(questions_labels)):
            q = questions_labels.iloc[i]
            q_text = nltk.word_tokenize(q['title'])
            if prediction_task == "category":
                label = q['topic']
            else:
                label = q['topic']

            if label is None or label == "":
                continue
            overall_counter[label] += 1

            if labels is not None and label not in labels:
                # print("Skipping label %s" % label)
                continue

            if counter[label] >= limit:
                continue

            # if (label not in {"Music", "Science", "History"}):
            #     continue
            tokens_labels.append((q_text, label))

        #     check if all labels have limit
            counter[label] += 1
            if labels is not None and np.all([counter[label] >= limit for label in labels]):
                break

        print("Overall Counter", overall_counter)
        print(counter)
        print("Loaded %d data from %s" % (len(tokens_labels), filename))
        return tokens_labels