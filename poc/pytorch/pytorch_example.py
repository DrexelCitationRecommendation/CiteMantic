import torch.nn as nn
from torch.autograd import Variable
from torch import optim, from_numpy
import numpy as np
import logging

FORMAT = "[%(asctime)s] - %(levelname)s - [%(funcName)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.output = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.output(out)
        return out


class WordVectors(object):
    def __init__(self, size=300):
        logging.info("Creating vector set...")
        file_path = r"C:\Users\eagle\Documents\Personal Projects\Python Projects\Data\Glove\glove.6B.{}d.txt".format(size)
        self.vocab = {}
        with open(file_path, 'r', encoding='utf8') as F:
            for line in F:
                word, v = line.split(' ', 1)
                vector = np.array([float(i) for i in v.split(' ')])
                self.vocab[word] = vector
        logging.info("Vector set created.")

    def in_vocab(self, word):
        return word in self.vocab.keys()


def train_model(word_set, word_vectors, model, epochs):
    logging.info("Training model...")
    X = []
    Y = []
    for word, y in word_set.items():
        if word_vectors.in_vocab(word):
            X.append(word_vectors.vocab[word])
            Y.append(y)
        else:
            logging.error(f"{word} not in vocab!")
            continue

    X = Variable(from_numpy(np.vstack(X))).float()
    y = Variable(from_numpy(np.vstack(Y))).float()

    loss_func = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)


    for e in range(epochs):
        output = model(X)
        loss = loss_func(output, y)
        logging.info(f"Epoch: {e+1}/{epochs}\t Loss: {loss}")
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_train(X, y, model, epochs):
    X = Variable(from_numpy(np.vstack(X))).long()
    y = Variable(from_numpy(np.vstack(y))).long()

    loss_func = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=0.1)
    for e in range(epochs):
        output = model(X)
        loss = loss_func(output, y)
        logging.info(f"Epoch: {e+1}/{epochs}\t Loss: {loss}")
        opt.zero_grad()
        loss.backward()
        opt.step()


def multi_split(line, split_chars, i=0, filter_empty=True):
    try:
        c = split_chars[i]
        if i == 0:
            return multi_split(line.split(c), split_chars, i+1)
        else:
            out = []
            for l in line:
                out.extend(l.split(c))
            return multi_split(out, split_chars, i+1)
    except IndexError:
        if filter_empty:
            return [l for l in line if l != ""]
        else:
            return line

def categorizer(result):
    result = [round(r) for r in result.tolist()]
    if result == [0, 1]:
        return "Not a Skill"
    elif result == [1,0]:
        return "Skill"


def main():
    # Simple Example: Learning "AND"

    # model = Model(input_size=2, hidden_size=4, output_size=2)
    # X = [[1,1], [1,0], [0,1], [0,0]]
    # y = [[0, 1], [1, 0], [1, 0], [1, 0]]
    # test_train(X, y, model, 25)


    # More complex example: Learning "skill words" from resumes

    wv = WordVectors()
    #test_data = {"Python": [0, 1], "test": [1, 0], "SQL": [0, 1], "parsnip": [1, 0]}
    test_data = {}
    with open("pytorch_example_data.csv", 'r') as F:
        for line in F:
            word, y, n = line.split(",")
            test_data[word] = [int(y), int(n)]

    model = Model(input_size=300, hidden_size=1000, output_size=2)

    train_model(test_data, wv, model, 25)

    text = open("ResumeText_JL.txt").read()
    words = [w.lower() for w in multi_split(text, [",", " ", "\n", "-"])]
    words = list(filter(wv.in_vocab, words))

    w_vec = [wv.vocab[word] for word in words]
    y_pred = model(Variable(from_numpy(np.vstack(w_vec))).float())
    preds = dict(zip(words, y_pred))
    for k, v in preds.items():
        print(k, "----->", categorizer(v))



if __name__ == "__main__":
    main()
