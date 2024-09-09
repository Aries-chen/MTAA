#!/user/bin/env python3
import os
import tensorflow as tf
import pandas as pd
from transformers import AutoTokenizer


def encode(texts, tokenizer, max_length=512):
    # Tokenize essays
    encodings = tokenizer(texts.tolist(), add_special_tokens=True, max_length=max_length,
                          return_attention_mask=True, return_tensors="np", truncation=True, padding='max_length')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    return input_ids, attention_mask


def multi_y(task_type, df):
    # Prepare multi-output scores
    if task_type == 'ell':
        df = df[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].astype('float32')
        df = df.cohesion, df.syntax, df.vocabulary, df.phraseology, df.grammar, df.conventions
    elif task_type == 'asap_12':
        df = df[['Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions']].astype('float32')
        df = df.Content, df.Organization, df['Word Choice'], df['Sentence Fluency'], df.Conventions

    elif task_type == 'asap_36':
        df = df[['Content', 'Prompt Adherence', 'Language', 'Narrativity']].astype('float32')
        df = df.Content, df['Prompt Adherence'], df.Language, df.Narrativity

    df = [i.to_numpy().reshape(-1, 1) for i in df]
    return tuple(df)


def get_df(task_type, train_data, test_data, tokenizer, text_type, batch_size):
    x_train = encode(train_data[text_type], tokenizer=tokenizer)
    x_test = encode(test_data[text_type], tokenizer=tokenizer)
    y_train_multi = multi_y(task_type, train_data)
    y_test_multi = multi_y(task_type, test_data)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_multi)).shuffle(len(x_train)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_multi)).batch(batch_size)
    return train_ds, test_ds


def get_y_true(task_type, test_data):
    if task_type == 'ell':
        y_true = test_data[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].to_numpy()
    elif task_type == 'asap_12':
        y_true = test_data[['Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions']].to_numpy()
    elif task_type == 'asap_36':
        y_true = test_data[['Content', 'Prompt Adherence', 'Language', 'Narrativity']].to_numpy()
    else:
        raise ValueError('Invalid task type')
    return y_true


def get_head_num(task_type):
    if task_type == 'ell':
        head_num = 6
    elif task_type == 'asap_12':
        head_num = 5
    elif task_type == 'asap_36':
        head_num = 4
    else:
        raise ValueError('Invalid task type')
    return head_num
