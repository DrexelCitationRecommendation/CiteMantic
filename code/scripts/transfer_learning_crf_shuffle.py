"""
Transfer learning for claim prediction using Discourse CRF model
"""
import sys
sys.path.insert(0, '.')
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

import random
random.seed(42)

EMBEDDING_DIM = 300
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.json'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.json'
TEST_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/test_labels.json'
# DISCOURSE_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/model_crf.tar.gz'
DISCOURSE_MODEL_PATH = './output_crf_pubmed_rct_glove/model.tar.gz'
archive = load_archive(DISCOURSE_MODEL_PATH)
discourse_predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')

# %%
# Re-implement Trainer dependencies
import logging
import math
import os
import time
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb, lazy_groups_of
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from allennlp.training import util as training_util
from allennlp.training.moving_average import MovingAverage

logger = logging.getLogger(__name__)

class ShuffleTrainer(Trainer):
    def __init__(
        self,
        dataset_reader: DatasetReader,
        train_dataset_path,
        model: Model,
        optimizer: torch.optim.Optimizer,
        iterator: DataIterator,
        train_dataset: Iterable[Instance],
        validation_dataset: Optional[Iterable[Instance]] = None,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: Union[int, List] = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None
    ) -> None:
        super(ShuffleTrainer, self).__init__(model=model,
                optimizer=optimizer,
                iterator=iterator,
                validation_iterator=iterator,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                patience=3,
                num_epochs=1,
                cuda_device=-1)
        self.dataset_reader = dataset_reader
        self.train_dataset_path = train_dataset_path
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches

        # Reset data
        # self.reader = self.dataset_reader()
        # self.train_data = self.reader.read(self.train_dataset_path)
        self.train_data = self.dataset_reader.read(self.train_dataset_path)

        # raw_train_generator = self.iterator(self.train_data, num_epochs=1, shuffle=self.shuffle)
        raw_train_generator = self.iterator(self.train_data, num_epochs=1, shuffle=False)
        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data) / num_gpus)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)
        cumulative_batch_size = 0
        for batch_group in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            loss = self.batch_loss(batch_group, for_training=True)

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss.backward()

            train_loss += loss.item()

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.named_parameters()
                }
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1))
                    param_norm = torch.norm(param.view(-1)).cpu()
                    self._tensorboard.add_train_scalar(
                        "gradient_update/" + name, update_norm / (param_norm + 1e-7)
                    )
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                cumulative_batch_size += cur_batch
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size / batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if self._model_save_interval is not None and (
                time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                    "{0}.{1}".format(epoch, training_util.time_to_str(int(last_save_time)))
                )
        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        return metrics

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

                if self.is_training:
                    # Shuffle data
                    zipped = list(zip(sents, labels))
                    random.shuffle(zipped)

                    sents, labels = zip(*zipped)
                    # print('Zipped:', zipped)
                print('Sentences:', sents)
                print('Labels:', labels)
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
    """
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
                                                            #   torch.nn.Linear(EMBEDDING_DIM, num_classes)])
    model.crf = ConditionalRandomField(num_classes, constraints, 
                                       include_start_end_transitions=include_start_end_transitions)
    # model.label_projection_layer = TimeDistributed(Linear(2 * EMBEDDING_DIM, num_classes))
    # print('Embedding dimension:', model.sentence_encoder.get_output_dim())
    model.label_projection_layer = TimeDistributed(Linear(model.sentence_encoder.get_output_dim(), num_classes))
    
    # model.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 2]))

    reader = ClaimAnnotationReaderJSON()
    # train_dataset = reader.read(TRAIN_PATH)
    train_dataset = reader.read('./scripts/fake_dataset.json')

    # validation_dataset = reader.read(VALIDATION_PATH)
    # test_dataset = reader.read(TEST_PATH)
    vocab = discourse_predictor._model.vocab
    vocab._token_to_index['labels'] = {'0': 0, '1': 1}

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    iterator = BasicIterator(batch_size=1)
    iterator.index_with(vocab)

    """Basic sanity check"""
    # iterating = iter(iterator(train_dataset, shuffle=False))
    # batch = next(iter(iterator(train_dataset, shuffle=False)))
    # batch = next(iterating)
    # tokens = batch["sentences"]
    # print('Zero epoch tokens:', tokens)
    # labels = batch["labels"]
    # print('Zero epoch labels:', labels)

    # for i in range(749):
    #     batch = next(iter(iterator(train_dataset, shuffle=False)))
    #     tokens = batch["sentences"]
    #     labels = batch["labels"]

    # batch = next(iter(iterator(train_dataset, shuffle=False)))
    # tokens = batch["sentences"]
    # print('One epoch tokens:', tokens)
    # labels = batch["labels"]
    # print('One epoch labels:', labels)

    print('Start training:')
    # unfreeze top layers and train
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
    trainer = ShuffleTrainer(
        dataset_reader=reader,
        train_dataset_path='./scripts/fake_dataset.json',
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        validation_iterator=iterator,
        train_dataset=train_dataset,
        # validation_dataset=validation_dataset,
        patience=3,
        num_epochs=1,
        cuda_device=-1
    )
    trainer.train()

    # reader = ClaimAnnotationReaderJSON()
    # train_dataset = reader.read(TRAIN_PATH)
    # train_dataset = reader.read('./scripts/fake_dataset.json')

    print('Start training 2:')
    # unfreeze top layers and train
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
    trainer = ShuffleTrainer(
        dataset_reader=reader,
        train_dataset_path='./scripts/fake_dataset.json',
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        validation_iterator=iterator,
        train_dataset=train_dataset,
        # validation_dataset=validation_dataset,
        patience=3,
        num_epochs=1,
        cuda_device=-1
    )
    trainer.train()

    # print('Start training 2:')
    # # unfreeze most layers and continue training
    # for param in list(model.parameters())[1:]:
    #     param.requires_grad = True
    # trainer = Trainer(
    #     model=model,
    #     optimizer=optimizer,
    #     iterator=iterator,
    #     validation_iterator=iterator,
    #     train_dataset=train_dataset,
    #     validation_dataset=validation_dataset,
    #     patience=3,
    #     num_epochs=1,
    #     cuda_device=-1
    # )
    # trainer.train()

    # # precision, recall, f-score on validation set
    # validation_list = read_json(cached_path(VALIDATION_PATH))
    # claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
    # y_pred, y_true = [], []
    # for val in validation_list:
    #     # print('val:', val)
    #     pred = claim_predictor.predict_json(val)
    #     logits = torch.FloatTensor(pred['logits'])
    #     best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0), 
    #                                         torch.LongTensor(pred['mask']).unsqueeze(0))
    #     predicted_labels = best_paths[0][0]
    #     y_pred.extend(predicted_labels)
    #     y_true.extend(val['labels'])
    # y_true = np.array(y_true).astype(int)
    # print('y_true shape:', y_true.shape)
    # y_pred = np.array(y_pred).astype(int)
    # print('y_pred shape:', y_pred.shape)
    # print('Validation score:', precision_recall_fscore_support(y_true, y_pred, average='binary'))

    # # precision, recall, f-score on test set
    # test_list = read_json(cached_path(TEST_PATH))
    # claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
    # y_pred, y_true = [], []
    # for tst in test_list:
    #     pred = claim_predictor.predict_json(tst)
    #     logits = torch.FloatTensor(pred['logits'])
    #     best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0), 
    #                                         torch.LongTensor(pred['mask']).unsqueeze(0))
    #     predicted_labels = best_paths[0][0]
    #     y_pred.extend(predicted_labels)
    #     y_true.extend(tst['labels'])
    # y_true = np.array(y_true).astype(int)
    # y_pred = np.array(y_pred).astype(int)
    # print('Test score:', precision_recall_fscore_support(y_true, y_pred, average='binary'))