import pandas as pd
import csv
with open('data/labelled_newscatcher_dataset.csv') as csv_file:
    reader = csv.DictReader(csv_file)
    data_list = []
    for row in reader:
        data_list.append(row)
    print(data_list[:5])

df = pd.read_csv('data/labelled_newscatcher_dataset.csv', delimiter=';')

# English entries only
df = df[df['lang'] == 'en']

print(f'Number of entries: {df.shape[0]}')

# Randomly Sample smaller portion of entries
n = 5_000
small_df = df.sample(n=5_000, random_state=42)
print(f'Number of sampled entries: {small_df.shape[0]}')
# print(f'Examples:')
# print(small_df[:5])

# Split into Train and Test sets
train_ratio = 0.70
dev_ratio = 0.20
train_n = round(small_df.shape[0]*train_ratio)
dev_n = round(small_df.shape[0]*dev_ratio)
train = small_df.sample(n=train_n, random_state=42)
remaining = small_df.drop(train.index)
dev = remaining.sample(n=dev_n, random_state=42)
test = remaining.drop(dev.index)
print(f'Number in training set: {train.shape[0]}')
print(f'Number in dev set: {dev.shape[0]}')
print(f'Number in test set: {test.shape[0]}')

small_df.to_csv(f'data/data_{n}.csv', index=False)
train.to_csv(f'data/data_{n}.train.csv', index=False)
dev.to_csv(f'data/data_{n}.dev.csv', index=False)
test.to_csv(f'data/data_{n}.test.csv', index=False)
