#!/user/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers
import transformers
import numpy as np


class MTAA(tf.keras.Model):
    def __init__(self, MODEL_PATH, cfg, model_type, task_type):
        super(MTAA, self).__init__()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.backbone = transformers.TFAutoModel.from_pretrained(MODEL_PATH, config=cfg)
        # self.backbone.trainable = False
        self.model_type = model_type
        self.mean_pool = MeanPool()
        self.oc = OC()
        self.task_type = task_type
        if self.task_type == 'ell':
            self.head_num = 6
        elif self.task_type == 'asap_12':
            self.head_num = 5
        elif self.task_type == 'asap_36':
            self.head_num = 4
        else:
            raise ValueError('Invalid task type')

        self.fc1_layers = [layers.Dense(512, name=f'y1{i}') for i in range(1, self.head_num+1)]
        self.fc2_layers = [layers.Dense(1, name=f'y2{i}') for i in range(1, self.head_num+1)]

    def call(self, inputs, y, **kwargs):
        input_ids = inputs[0]
        attention_masks = inputs[1]
        model_type = self.model_type
        if model_type == 'deberta':
            output = self.backbone.deberta(input_ids, attention_mask=attention_masks)

        x = output.last_hidden_state
        x = self.mean_pool(x, mask=attention_masks)

        y_list = []
        for i in range(self.head_num):
            y1 = self.fc1_layers[i](x)
            y2 = self.fc2_layers[i](y1)
            y_list.append(y2)

            vanilla_loss = self.loss_fn(y[i], y2)
            oc_loss = self.oc(x, y2)
            task_loss = vanilla_loss + oc_loss
            self.add_loss(task_loss)

        return y_list


def extract_task_weights(model, task_index):
    trainable_variables_dict = {variable.name: variable for variable in model.trainable_variables}

    y1_weights = trainable_variables_dict.get(f"mtaa/y1{task_index}/kernel:0")
    y1_bias = trainable_variables_dict.get(f"mtaa/y1{task_index}/bias:0")
    y2_weights = trainable_variables_dict.get(f"mtaa/y2{task_index}/kernel:0")
    y2_bias = trainable_variables_dict.get(f"mtaa/y2{task_index}/bias:0")

    return y1_weights, y1_bias, y2_weights, y2_bias


def adjust_weights(loss_rates, alpha):
    weights = np.array(loss_rates) ** alpha
    M = np.sum(weights) / len(loss_rates)
    task_weights = weights / M

    return task_weights


class MeanPool(tf.keras.layers.Layer):
    def call(self, inputs, mask=None):
        expanded_mask = tf.cast(tf.expand_dims(mask, -1), dtype="float32")
        masked_embeddings_sum = tf.reduce_sum(inputs * expanded_mask, axis=1)
        mask_sum = tf.reduce_sum(expanded_mask, axis=1)
        mask_sum = tf.maximum(mask_sum, tf.constant([1e-9]))
        pooled_embeddings = masked_embeddings_sum / mask_sum
        return pooled_embeddings


def model_selection(model_type, dr):
    if model_type == 'deberta':
        MODEL_PATH = r"./pretrained_models/deberta-v3-base/"
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)  # 本地加载
    else:
        raise ValueError('Invalid model type')

    cfg = transformers.AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    cfg.hidden_dropout_prob = dr
    cfg.attention_probs_dropout_prob = dr
    cfg.save_pretrained(MODEL_PATH + '/tokenizer/')
    return MODEL_PATH, tokenizer, cfg


class OC(tf.keras.layers.Layer):
    def call(self, shared_features, task_features, regularization_factor=0.01):
        tf.debugging.check_numerics(shared_features, "shared_features has NaN or Inf")
        tf.debugging.check_numerics(task_features, "task_features has NaN or Inf")

        # Orthogonality Constraints
        # Modified from https://github.com/FrankWork/fudan_mtl_reviews/blob/master/src/models/mtl_model.py
        shared_mean = tf.reduce_mean(shared_features, axis=0)
        task_mean = tf.reduce_mean(task_features, axis=0)

        centered_shared_features = shared_features - shared_mean
        centered_task_features = task_features - task_mean

        normalized_shared = tf.nn.l2_normalize(centered_shared_features, axis=1)
        normalized_task = tf.nn.l2_normalize(centered_task_features, axis=1)

        correlation_matrix = tf.matmul(normalized_task, normalized_shared, transpose_a=True)

        cost = tf.reduce_mean(tf.square(correlation_matrix)) * regularization_factor
        cost = tf.maximum(cost, 0)

        assert_op = tf.debugging.assert_all_finite(cost, 'Non-finite cost')
        with tf.control_dependencies([assert_op]):
            loss = tf.identity(cost)

        return loss
