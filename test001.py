import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input
import transformers
from nltk.corpus import stopwords
import re
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
import re
from collections import Counter 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from keras.preprocessing.sequence import pad_sequences


def clean(s):

    '''
    CLean the text data
    '''
    replace_by_space = re.compile(r'[\n/(){}\[\]\|@,;]')
    STOPWORDS = set(stopwords.words('english'))
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z !]')

    s = s.lower()
    s = replace_by_space.sub(' ',s)
    s = BAD_SYMBOLS_RE.sub('', s)
    s = ' '.join(word for word in s.split() if word not in STOPWORDS) # delete stopwors from text
    return s


def tokenizer(df, MAX_LEN = 40):

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenized = df["Text"].apply(lambda s: tokenizer.encode(s,add_special_tokens=True))
    input_ids = pad_sequences(tokenized, maxlen=MAX_LEN, dtype="long", 
                          value=0, truncating="post", padding="post")

    return input_ids



def build_models(max_len=40):

    '''
    outputs three models for each category
    '''
    
    transformer =  transformers.TFBertModel.from_pretrained('bert-base-uncased')
    
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    models = [Model(inputs=input_word_ids, outputs=out), 
              Model(inputs=input_word_ids, outputs=out), Model(inputs=input_word_ids, outputs=out)]
    for model in models:
        model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['AUC'])
    
    models[2].load_weights("checkpoint_compliment.h5")

    return {"complaint": models[0], "suggestion": models[1], "compliment":models[2] }

    

def make_preds(model, x_test):
    
    pred = model.predict(x_test)
    
    return pred


def predict(df):
    
    df["Text"] = df["Text"].apply(lambda s: clean(s))
    input_ids = tokenizer(df, MAX_LEN = 40)

    models = build_models(max_len=40)
    y_pred = make_preds(models["compliment"], input_ids)

    y_pred_df = pd.DataFrame({'compliment': y_pred[:,0]})
    y_pred_df = pd.concat([df[["Text"]].reset_index(drop=True),y_pred_df.applymap(lambda n:round(n,2))], axis=1)
    
    return y_pred_df