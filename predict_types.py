import string
import re
import transformers
import tensorflow as tf
from scipy import stats
import pandas as pd
import numpy as np
from transformers import TFBertModel, BertTokenizer
import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow.keras.layers as layers



maxlen = 1500
per_types = ['ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP','INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISTJ','ISTP']
N_AXIS = 4
MAX_SEQ_LEN = 128
BERT_NAME = 'bert-base-uncased'
axes = ["I-E","N-S","T-F","J-P"]
classes = {"I":0, "E":1, # axis 1
           "N":0,"S":1, # axis 2
           "T":0, "F":1, # axis 3
           "J":0,"P":1} # axis 4

def text_preprocessing(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]
    return text




def prepare_bert_input(sentences, seq_len, bert_name):
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    encodings = tokenizer(sentences.tolist(), truncation=True, padding='max_length',
                                max_length=seq_len)
    input = [np.array(encodings["input_ids"]), np.array(encodings["token_type_ids"]),
               np.array(encodings["attention_mask"])]
    return input
    

def recreate_model(): 
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
    input_type = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
    input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_type, input_mask]
    bert = TFBertModel.from_pretrained(BERT_NAME)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state
    avg = tf.keras.layers.GlobalAveragePooling1D()(last_hidden_states)
    output = tf.keras.layers.Dense(N_AXIS, activation="sigmoid")(avg)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.00001), metrics=[keras.metrics.AUC(multi_label=True, curve="ROC"),
                                                  keras.metrics.BinaryAccuracy()])
    model.load_weights("bert_base_model1.h5")
    return model

new_model = recreate_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_type(text):
    sentences = np.asarray([text])
    enc_sentences = prepare_bert_input(sentences, MAX_SEQ_LEN, BERT_NAME)
    predictions = new_model.predict(enc_sentences)
    for sentence, pred in zip(sentences, predictions):
        pred_axis = []
        mask = (pred > 0.5).astype(bool)
        for i in range(len(mask)):
            if mask[i]:
                pred_axis.append(axes[i][2])
            else:
                pred_axis.append(axes[i][0])
        print('-- comment: '+sentence.replace("\n", "").strip() +
          '\n-- personality: '+str(pred_axis) +
          '\n-- scores:'+str(pred))
    result = ''.join(pred_axis)
    return result

def predict_tweet(result):
    op_json = {}
    op_json["name"] = str(username)
    op_json["type"] = result
    info_df = pd.read_csv("reference_data/MBTI.csv")
    op_json["traits"] = info_df[info_df["type"]==per_op]["traits"].values[0]
    op_json["career"] = info_df[info_df["type"]==per_op]["career"].values[0]
    op_json["people"] = info_df[info_df["type"]==per_op]["eminent personalities"].values[0]
    op_json["per_name"] = info_df[info_df["type"]==per_op]["name"].values[0]
    
