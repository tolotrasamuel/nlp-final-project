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
    def load_and_tokenize_json_data(filename: str, limit: Optional[int] = None) -> list[TokenLabel]:
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
    
    @staticmethod
    def load_and_tokenize_csv_data(filename: str, limit: Optional[int] = None) -> list[TokenLabel]:
        print("Loading data from %s" % filename)

        tokens_labels: list[TokenLabel] = []
        dataset = FileHelper.read_csv_file(filename)
        # label is index 0, question is index 4
        question_labels = dataset[1:]
        all_count = len(question_labels)
        if limit is not None and limit > 0:
            question_labels = question_labels[:limit]

        print(f"Tokenizing {len(question_labels)} out of {all_count} questions")


        exclude_labels = []
        include_labels = []
        for q in question_labels:
            if q[0] in exclude_labels:
                continue
            if len(include_labels) == 0 or q[0] in include_labels:
                tokens_labels.append((nltk.word_tokenize(q[4]), q[0]))

        print("Loaded %d data from %s" % (len(tokens_labels), filename))
        return tokens_labels