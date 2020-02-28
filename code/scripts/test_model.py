# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Change directory to VSCode workspace root so that relative path loads work correctly. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..'))
	print(os.getcwd())
except:
	pass


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

np.random.seed(42)

EMBEDDING_DIM = 300
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.json'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.json'
TEST_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/test_labels.json'
# DISCOURSE_MODEL_PATH = './output_crf_pubmed_rct_glove/model.tar.gz'
# archive = load_archive(DISCOURSE_MODEL_PATH)
# discourse_predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')


# %%
import random
random.seed(42)

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
                 lazy: bool = False,
                 is_training: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.is_training = is_training

    # @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                example = json.loads(line)
                sents = example['sentences']
                labels = example['labels']

                # if self.is_training:
                #     # Shuffle data
                #     zipped = list(zip(sents, labels))
                #     random.shuffle(zipped)

                #     sents, labels = zip(*zipped)
                #     print('Zipped:', zipped)

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
    pretrained_model="bert-base-uncased",
    max_pieces=300,
    do_lowercase=True,
 )


# %%
reader = ClaimAnnotationReaderJSON(
    token_indexers={"tokens": token_indexer}
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

iterator = BasicIterator(batch_size=64)

iterator.index_with(vocab)


# %%
def multiple_target_CrossEntropyLoss(logits, labels):
    loss = 0
    for i in range(logits.shape[0]):
        loss = loss + nn.CrossEntropyLoss()(logits[i, :, :], labels[i, :])
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
        embedded_sentence = self.text_field_embedder(sentences)
        sentence_mask = util.get_text_field_mask(sentences)
        encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask)

        logits = self.classifier_feedforward(encoded_sentence)
        # logits = logits.squeeze(-1) # Added to squeeze 3d to 2d

        output_dict = {'logits': logits}
        if labels is not None:
            print("label shape:", labels.shape)
            print("logits shape:", logits.shape)
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
        for instance_labels in output_dict["logits"]:
            print('Instance labels:', instance_labels)
        # output_dict["labels"] = [
        #     [self.vocab.get_token_from_index(label, namespace='labels')
        #          for label in instance_labels]
        #         for instance_labels in output_dict["logits"]
        # ]
        output_dict["labels"] = [
            [np.argmax(label.cpu().data.numpy()) for label in instance_labels]
                for instance_labels in output_dict["logits"]
        ]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


# %%
"""Prepare embeddings"""
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

bert_embedder = PretrainedBertEmbedder(
    pretrained_model = "bert-base-uncased",
    top_layer_only=True,
    requires_grad=False
)

word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder(
                                                            token_embedders={"tokens": bert_embedder}, 
                                                            allow_unmatched_keys=True)


# %%
BERT_DIM = word_embeddings.get_output_dim()
print('BERT dim:', BERT_DIM)

class BertSentencePooler(Seq2VecEncoder):
    def forward(self, embs:torch.tensor, mask:torch.tensor=None) -> torch.tensor:
        return embs[:, :, 0]
    
    def get_output_dim(self) -> int:
        return BERT_DIM

sentence_encoder = BertSentencePooler(vocab)


# %%
classifier_feedforward = nn.Linear(BERT_DIM, 2)

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
# def multiple_target_CrossEntropyLoss(logits, labels):
#     loss = 0
#     for i in range(logits.shape[0]):
#         loss = loss + nn.CrossEntropyLoss()(logits[i, :, :], labels[i, :])
#     return loss


# %%
# loss.backward()


# %%
# loss = model(**batch)["loss"]


# %%
"""Train"""
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    validation_iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    patience=5,
    num_epochs=1,
    cuda_device=-1
)


# %%
print('Start training')
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
test_list = read_json(cached_path(TEST_PATH))
claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
y_pred, y_true = [], []
for tst in test_dataset:
    print('tst', tst)
    pred = claim_predictor.predict_instance(tst)
    print('Pred output:', pred)
    logits = torch.FloatTensor(pred['logits'])
    print(logits.shape)
    print('Logits output:', logits)
#     best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0), 
#                                         torch.LongTensor(pred['mask']).unsqueeze(0))
    predicted_labels = pred['labels']
    y_pred.extend(predicted_labels)
    y_true.extend(tst['labels'])
    break
y_true = np.array(y_true).astype(int)
y_pred = np.array(y_pred).astype(int)
print('Y true:', y_true)
print('Y pred:', y_pred)
print('Test score:', precision_recall_fscore_support(y_true, y_pred, average='binary'))


# %%


