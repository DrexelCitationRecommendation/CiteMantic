"""
Transfer learning for claim prediction using Discourse CRF model
"""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '/Users/kchu/Documents/Projects/Senior Project/Claim Extraction/detecting-scientific-claim-master/')

import os
cwd = os.getcwd()
print('Current directory:', cwd)

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

from discourse import read_json
from discourse.predictors import DiscourseClassifierPredictor
from discourse.dataset_readers import ClaimAnnotationReaderJSON
from discourse.models import DiscourseClassifier

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common import Params
from allennlp.common.file_utils import cached_path

from allennlp.data.fields import Field, TextField, LabelField, ListField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler

from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward
from torch.nn.modules.linear import Linear
import torch.nn as nn


EMBEDDING_DIM = 300
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.json'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.json'
TEST_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/test_labels.json'
# DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz'
# DISCOURSE_MODEL_PATH = '../output_crf_pubmed_rct_glove/model.tar.gz' # on the server
DISCOURSE_MODEL_PATH = './output_crf_pubmed_rct_glove/model.tar.gz' # on the local machine
archive = load_archive(DISCOURSE_MODEL_PATH)
discourse_predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')

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
                 is_validating: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.is_validating = is_validating

    # @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                example = json.loads(line)
                sents = example['sentences']
                labels = example['labels']

                if self.is_validating:
                    # Shuffle 50% data
                    choice = random.choice([0, 1])
                    if choice == 0:
                        # Shuffle data
                        zipped = list(zip(sents, labels))
                        random.shuffle(zipped)
                        sents, labels = zip(*zipped)
                    # print('Zipped:', zipped)

                # print('Sentences:', sents)
                # print('Labels:', labels)
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
class ClaimCrfPredictor(Predictor):
    """"
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance


if __name__ == '__main__':
    # load model and freeze all layers
    model = discourse_predictor._model
    for param in list(model.parameters()):
        param.requires_grad = False
    num_classes, constraints, include_start_end_transitions = 2, None, False
    # model.classifier_feedforward._linear_layers = ModuleList([torch.nn.Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM), 
    #                                                           torch.nn.Linear(EMBEDDING_DIM, num_classes)])
    sentence_encoder = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=EMBEDDING_DIM, bidirectional=True, num_layers=2)
    model.sentence_encoder = sentence_encoder
    model.crf = ConditionalRandomField(num_classes, constraints, 
                                       include_start_end_transitions=include_start_end_transitions)
    model.label_projection_layer = TimeDistributed(Linear(2 * EMBEDDING_DIM, num_classes))

    reader = ClaimAnnotationReaderJSON()
    validate_reader = ClaimAnnotationReaderJSON(is_validating=True)
    # train_dataset = reader.read(TRAIN_PATH)
    train_dataset = validate_reader.read(TRAIN_PATH)
    # validation_dataset = reader.read(VALIDATION_PATH)
    validation_dataset = validate_reader.read(VALIDATION_PATH)
    test_dataset = reader.read(TEST_PATH)
    vocab = discourse_predictor._model.vocab
    vocab._token_to_index['labels'] = {'0': 0, '1': 1}

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    iterator = BasicIterator(batch_size=64)
    iterator.index_with(vocab)

    # model.cuda(0) # server only
    
    print('Start training 1:')
    # unfreeze top layers and train
    # for param in list(model.parameters())[:-4]:
    #     param.requires_grad = False
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        validation_iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        patience=3,
        num_epochs=1, 
        cuda_device=-1
    )
    trainer.train()

    # precision, recall, f-score on validation set
    validation_list = read_json(cached_path(VALIDATION_PATH))
    claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
    y_pred, y_true = [], []
    for val in validation_list:
        pred = claim_predictor.predict_json(val)
        logits = torch.FloatTensor(pred['logits'])
        best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0), 
                                            torch.LongTensor(pred['mask']).unsqueeze(0))
        predicted_labels = best_paths[0][0]
        y_pred.extend(predicted_labels)
        y_true.extend(val['labels'])
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    print(precision_recall_fscore_support(y_true, y_pred, average='binary'))

    # precision, recall, f-score on test set
    test_list = read_json(cached_path(TEST_PATH))
    claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
    y_pred, y_true = [], []
    for tst in test_list:
        pred = claim_predictor.predict_json(tst)
        logits = torch.FloatTensor(pred['logits'])
        best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0),
                                            torch.LongTensor(pred['mask']).unsqueeze(0))
        predicted_labels = best_paths[0][0]
        y_pred.extend(predicted_labels)
        y_true.extend(tst['labels'])
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    print('Test score:', precision_recall_fscore_support(y_true, y_pred, average='binary'))