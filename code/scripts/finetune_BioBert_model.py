"""
Transfer learning for claim prediction using Discourse CRF model
"""
# %%
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '/Users/kchu/Documents/Projects/Senior Project/Claim Extraction/detecting-scientific-claim-master/')

from typing import Iterator, List, Dict, Optional
import os
import json
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from allennlp.common.util import JsonDict

import torch
import torch.optim as optim
from torch.nn import ModuleList
import torch.nn.functional as F

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common import Params
from allennlp.common.file_utils import cached_path

from allennlp.data.fields import Field, TextField, LabelField, ListField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import PretrainedBertIndexer

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
import torch.nn as nn

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler

from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward
from torch.nn.modules.linear import Linear


EMBEDDING_DIM = 300
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.json'
# TRAIN_PATH = './train_augmented_labels.json'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.json'
TEST_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/test_labels.json'
# DISCOURSE_MODEL_PATH = './output_crf_pubmed_rct_glove/model.tar.gz'
# archive = load_archive(DISCOURSE_MODEL_PATH)
# discourse_predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')

# %%
class ClaimAnnotationReaderJSON(DatasetReader):
    """
    Reading annotation dataset in the following JSON format:

    {
        "paper_id": ..., 
        "user_id": ...,
        "sentences": [..., ..., ...],
        "labels": [..., ..., ...] 
    }
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    # @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                example = json.loads(line)
                sents = example['sentences']
                labels = example['labels']
                yield self.text_to_instance(sents, labels)

    # @overrides
    def text_to_instance(self,
                         sents: List[str],
                         labels: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_sents = [self._tokenizer.tokenize(sent) for sent in sents]
        sentence_sequence = ListField([TextField(tk, self._token_indexers) for tk in tokenized_sents])
        fields['sentences'] = sentence_sequence
        
        if labels is not None:
            fields['labels'] = SequenceLabelField(labels, sentence_sequence)
        return Instance(fields)

# %%
token_indexer = PretrainedBertIndexer(
    pretrained_model="./biobert_v1.1_pubmed/vocab.txt",
    do_lowercase=True,
 )

# %%
reader = ClaimAnnotationReaderJSON(
    token_indexers={"tokens": token_indexer},
    lazy=True
)

train_dataset = reader.read(TRAIN_PATH)
validation_dataset = reader.read(VALIDATION_PATH)
test_dataset = reader.read(TEST_PATH)
# %%
vocab = Vocabulary()

vocab._token_to_index['labels'] = {'0': 0, '1': 1}

# %%
"""Prepare iterator"""
from allennlp.data.iterators import BasicIterator

iterator = BasicIterator(batch_size=8)

iterator.index_with(vocab)

# %%
def multiple_target_CrossEntropyLoss(logits, labels):
    loss = 0
    for i in range(logits.shape[0]):
        loss = loss + nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0]).cuda())(logits[i, :, :], labels[i, :])
    return loss

# %%
"""Prepare the model"""
class BaselineModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BaselineModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sentence_encoder = sentence_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                sentences: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # print('Sentences:', sentences['tokens'].size())
        embedded_sentence = self.text_field_embedder(sentences)
        # print('Embedded size:', embedded_sentence.size())
        sentence_mask = util.get_text_field_mask(sentences)
        # print('Sentence mask:', sentence_mask.size())
        encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask)
        # print('Encoded sentence:', encoded_sentence.size())

        logits = self.classifier_feedforward(encoded_sentence)
        logits = logits.squeeze(-1) # Added to squeeze 3d to 2d

        output_dict = {'logits': logits}
        if labels is not None:
            # print("label shape:", labels.shape)
            # print("logits shape:", logits.shape)
            # loss = self.loss(logits, labels.squeeze(-1))
            loss = multiple_target_CrossEntropyLoss(logits, labels)
            for metric in self.metrics.values():
                metric(logits, labels.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Coverts tag ids to actual tags.
        """
        # for instance_labels in output_dict["logits"]:
            # print('Instance labels:', instance_labels)
        # output_dict["labels"] = [
        #     [self.vocab.get_token_from_index(label, namespace='labels')
        #          for label in instance_labels]
        #         for instance_labels in output_dict["logits"]
        # ]
        output_dict["labels"] = [
            [np.argmax(label.cpu().data.numpy()) for label in instance_labels]
                for instance_labels in output_dict["logits"]
        ]
        # print(output_dict["logits"])
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

# %%
"""Prepare embeddings"""
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

bert_embedder = PretrainedBertEmbedder(
    pretrained_model = "./biobert_v1.1_pubmed/weights.tar.gz",
    top_layer_only=True,
    requires_grad=False
)

#print('Bert Model:', bert_embedder.bert_model.encoder.layer[11])
for param in bert_embedder.bert_model.encoder.layer[8:].parameters():
    param.requires_grad = True


word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder(
                                                            token_embedders={"tokens": bert_embedder}, 
                                                            allow_unmatched_keys=True)

# %%
BERT_DIM = word_embeddings.get_output_dim()
print('Bert dim:', BERT_DIM)

class BertSentencePooler(Seq2VecEncoder):
    def __init__(self, vocab):
        super().__init__(vocab)
        # self.lstm = nn.LSTM(input_size=BERT_DIM, hidden_size=128, num_layers=1, dropout=0.2, bidirectional=True)

    def forward(self, embs:torch.tensor, mask:torch.tensor=None) -> torch.tensor:
        bert_out = embs[:, :, 0]
        # print('Bert output shape:', embs.size())
        # return self.lstm(bert_out)[0]
        return bert_out
    
    def get_output_dim(self) -> int:
        return BERT_DIM

sentence_encoder = BertSentencePooler(vocab)

# %%
# classifier_feedforward = nn.Linear(256, 2)
classifier_feedforward = nn.Linear(768,2)

# %%
model = BaselineModel(
    vocab,
    word_embeddings,
    sentence_encoder,
    classifier_feedforward
)

# %%
"""Basic sanity check"""
batch = next(iter(iterator(train_dataset)))
tokens = batch["sentences"]
labels = batch["labels"]

# %%
import allennlp.nn.util as util

# mask = util.get_text_field_mask(tokens)

# %%
# embeddings = model.text_field_embedder(tokens)

# %%
# state = model.sentence_encoder(embeddings, mask)

# %%
# logits = model.classifier_feedforward(state)
# logits = logits.squeeze(-1)

# %%
# loss =  nn.NLLLoss()(logits.reshape(-1, 10), labels.reshape(-1, 10))
#def multiple_target_CrossEntropyLoss(logits, labels):
    # loss = 0
    # for i in range(logits.shape[0]):
        # loss = loss + nn.CrossEntropyLoss(weight=torch.tensor([1,3]))(logits[i, :, :], labels[i, :])
    # return loss

# %%
# loss.backward()

# %%
# loss = model(**batch)["loss"]

# %%
"""Train"""
# print('Parameters:', model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer)
optimizer = optim.SGD([{'params': model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer[11].parameters(), 'lr': 0.001},
                        {'params': model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer[10].parameters(), 'lr': 0.00095},
                        {'params': model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer[9].parameters(), 'lr': 0.0009},
                        {'params': model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer[8].parameters(), 'lr': 0.000855}
                        ], lr=0.001)
# Default
# optimizer = optim.SGD(model.parameters(), lr=0.001)

model = model.cuda()

print('Start training')

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    validation_iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    patience=3,
    num_epochs=50,
    cuda_device=[0, 1]
)

# %%
metrics = trainer.train()

# %%
"""Testing"""

class ClaimCrfPredictor(Predictor):
    """
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance

def read_json(file_path):
    """
    Read list from JSON path
    """
    if not os.path.exists(file_path):
        return []
    else:
        with open(file_path, 'r') as fp:
            ls = [json.loads(line) for line in fp]
        return ls

# %%
test_list = read_json(cached_path(VALIDATION_PATH))
claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
y_pred, y_true = [], []
for tst in validation_dataset:
    # print('tst', tst)
    pred = claim_predictor.predict_instance(tst)
    # print('Pred output:', pred)
    logits = torch.FloatTensor(pred['logits'])
    # print(logits.shape)
    # print('Logits output:', logits)
#     best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0),
#                                         torch.LongTensor(pred['mask']).unsqueeze(0))
    predicted_labels = pred['labels']
    y_pred.extend(predicted_labels)
    y_true.extend(tst['labels'])
    # break
y_true = np.array(y_true).astype(int)
y_pred = np.array(y_pred).astype(int)
# print('Y true:', y_true)
# print('Y pred:', y_pred)
print('Val score:', precision_recall_fscore_support(y_true, y_pred, average='binary'))
# Save y_true and y_pred
df = pd.DataFrame()
df['y_true'] = y_true
df['y_pred'] = y_pred
df.to_csv('biobert_y_true_pred_val.csv', index=False)

# %%
test_list = read_json(cached_path(TEST_PATH))
claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
y_pred, y_true = [], []
for tst in test_dataset:
    # print('tst', tst)
    pred = claim_predictor.predict_instance(tst)
    # print('Pred output:', pred)
    logits = torch.FloatTensor(pred['logits'])
    # print(logits.shape)
    # print('Logits output:', logits)
#     best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0),
#                                         torch.LongTensor(pred['mask']).unsqueeze(0))
    predicted_labels = pred['labels']
    y_pred.extend(predicted_labels)
    y_true.extend(tst['labels'])
    # break
y_true = np.array(y_true).astype(int)
y_pred = np.array(y_pred).astype(int)
# print('Y true:', y_true)
# print('Y pred:', y_pred)
print('Test score:', precision_recall_fscore_support(y_true, y_pred, average='binary'))

# Save model
with open(f"./finetune_model.th", "wb") as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files(f"./finetune_vocab.txt")

# Save y_true and y_pred
df = pd.DataFrame()
df['y_true'] = y_true
df['y_pred'] = y_pred
df.to_csv('biobert_y_true_pred_test.csv', index=False)