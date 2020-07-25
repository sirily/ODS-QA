import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text # Not used directly but needed to import TF ops.

from tensorflow.keras.layers import Input, Dot, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import transformers
from typing import List, Dict, Tuple, Iterable, Type

### TF
def compile_model(path_to_model, loss_fn, opt_fn):
    question = Input(shape=(1,), dtype=tf.string, name='Question')
    answer = Input(shape=(1,), dtype=tf.string, name='Answer')
    use = hub.KerasLayer(path_to_model, trainable=True, name='USE')
    #encode questions and answers
    q_emb = use(tf.squeeze(tf.cast(question, tf.string)))
    a_emb = use(tf.squeeze(tf.cast(answer, tf.string)))
    #apply cosine similarity function
    dist_matrix = 1 - tf.matmul(tf.math.l2_normalize(q_emb, axis=1), tf.math.l2_normalize(tf.transpose(a_emb), axis=0))
    model = Model(inputs=[question, answer], outputs=dist_matrix)
    model.compile(loss=loss_fn, optimizer=opt_fn, run_eagerly=True)

    return model

def train(path_to_model, loss_fn, opt_fn, X, y, batch_size, epochs):
    model = compile_model(path_to_model, loss_fn, opt_fn)
    hist = model.fit(X, y,
                    batch_size=batch_size, epochs=epochs)

    return model, hist

### PyTorch
def batch_to_device(batch, target_device):
    """
    send a batch to a device
    :param batch:
    :param target_device:
    :return: the batch sent to the device
    """
    features = batch['features']
    for paired_sentence_idx in range(len(features)):
        for feature_name in features[paired_sentence_idx]:
            features[paired_sentence_idx][feature_name] = features[paired_sentence_idx][feature_name].to(target_device)
    labels = batch['labels'].to(target_device)
    return features, labels

class SBERT(SentenceTransformer):
    def __init__(self, model_name_or_path: str = None, device: str = None):
        super().__init__(model_name_or_path, device=device)
        print(f'Using device {self.device}')

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0][0])

        labels = []
        paired_texts = [[] for _ in range(num_texts)]
        max_seq_len = [0] * num_texts
        for tokens, label in batch:
            labels.append(label)
            for i in range(num_texts):
                paired_texts[i].append(tokens[i])
                max_seq_len[i] = max(max_seq_len[i], len(tokens[i]))

        features = []
        for idx in range(num_texts):
            max_len = max_seq_len[idx]
            feature_lists = {}

            for text in paired_texts[idx]:
                sentence_features = self.get_sentence_features(text, max_len)

                for feature_name in sentence_features:
                    if feature_name not in feature_lists:
                        feature_lists[feature_name] = []

                    feature_lists[feature_name].append(sentence_features[feature_name])


            for feature_name in feature_lists:
                feature_lists[feature_name] = torch.cat(feature_lists[feature_name])

            features.append(feature_lists)

        return {'features': features, 'labels': torch.stack(labels)}

    def fit(self,
            train_objective: Iterable[Tuple[torch.utils.data.DataLoader, torch.nn.Module]],
            #evaluator: SentenceEvaluator,
            epoches: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[torch.optim.Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object ]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            local_rank: int = -1
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param weight_decay:
        :param scheduler:
        :param warmup_steps:
        :param optimizer:
        :param evaluation_steps:
        :param output_path:
        :param save_best_model:
        :param max_grad_norm:
        :param local_rank:
        :param train_objectives:
            Tuples of DataLoader and LossConfig
        :param evaluator:
        :param epochs:
        :param steps_per_epoch: Train for x steps in each epoch. If set to None, the length of the dataset will be used
        """
        train_loader, loss_model = train_objective

        param_optimizer = list(self.named_parameters())

        #init optimizer and sheduler
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = len(train_loader)

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = int(steps_per_epoch * epoches)
        if local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=t_total)

        train_loader.collate_fn = self.smart_batching_collate
        
        for ep in range(1, epoches+1):
            self.train()
            iterator = tqdm(train_loader, desc=f'Epoch {ep}', total=len(train_loader))
            batch_loss = []
            for batch in iterator:
                self.zero_grad()

                batch = batch_to_device(batch, self.device)

                #get embeddings
                seqs = [self(sentence_feature)['sentence_embedding'] for sentence_feature in batch[0]]
                seq_a, seq_b = seqs
                #pass them to the loss function
                loss_value = loss_model(seq_a, seq_b, batch[1])
                loss_value.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                batch_loss.append(loss_value.item())
                iterator.desc = f'Epoch {ep} Loss {sum(batch_loss)/len(batch_loss):.3}'


class SentencesDataset(Dataset):
    """
    Dataset for smart batching, that is each batch is only padded to its longest sequence instead of padding all
    sequences to the max length.
    The SentenceBertEncoder.smart_batching_collate is required for this to work.
    SmartBatchingDataset does *not* work without it.
    """
    def __init__(self, features: np.array, target: np.array, model: SBERT, show_progress_bar: bool = True):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor
        """
        self.show_progress_bar = show_progress_bar
        
        self.convert_input_examples(features, target, model)

    def convert_input_examples(self, texts: np.array, targets: np.array, model: SBERT):
        """
        Converts input examples to a SmartBatchingDataset usable to train the model with
        SentenceTransformer.smart_batching_collate as the collate_fn for the DataLoader
        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.
        :param examples:
            the input examples for the training
        :param model
            the Sentence BERT model for the conversion
        :return: a SmartBatchingDataset usable to train the model with SentenceTransformer.smart_batching_collate as the collate_fn
            for the DataLoader
        """
        num_texts = len(texts)
        inputs = []
        too_long = [0] * num_texts
        iterator = texts
        max_len = model.get_max_seq_length()

        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert dataset")

        for text in iterator:
            tokenized_texts = model.tokenize(text)
            """
            padding_length = max_len - len(tokenized_texts)
            if padding_length > 0:
                tokenized_texts = tokenized_texts + ([0] * padding_length)
            """
            inputs.append(tokenized_texts)    

        print('')
        print("Sentences longer than max_sequence_length: {}".format(sum(too_long)))

        self.tokens = inputs
        self.labels = targets

    def __getitem__(self, item):
        return self.tokens[item], self.labels[item]

    def __len__(self):
        return len(self.tokens)