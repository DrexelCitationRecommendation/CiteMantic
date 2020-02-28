from eda import eda, eda_4
import json
import os

# print(os.listdir())
train_path = './scripts/train_labels.json'
train_augmented_path = './scripts/train_augmented_4_labels.json'
train_augmented_abstracts = []

with open(train_path, 'r') as file:
    for line in file:
        # print(line)
        example = json.loads(line)
        sents = example['sentences']
        labels = example['labels']
        augmented_sents = []
        for sent in sents:
            if sent == '.':
                augmented_sent = ['.'] * 10
            else:
                augmented_sent = eda_4(sent)
            # if len(augmented_sent) != 10:
            #     print('Hey')
            augmented_sents.append(augmented_sent)

        for i in range(10):
            augmented_abstract = {}
            augmented_senteces = []
            for sent in augmented_sents:
                augmented_senteces.append(sent[i])
            augmented_abstract['sentences'] = augmented_senteces
            augmented_abstract['labels'] = labels
            train_augmented_abstracts.append(augmented_abstract)

print(len(train_augmented_abstracts))
# print(train_augmented_abstracts[9])
with open(train_augmented_path, 'w') as outfile:
    for abstract in train_augmented_abstracts:
        json.dump(abstract, outfile)
        outfile.write('\n')