import pandas as pd
from nltk.corpus import stopwords
import re
from collections import Counter 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from joblib import load


# df = pd.read_csv("./feedback.csv")


#### some functions to preprocess the text data

def clean(s):

    '''
    CLean the text data
    '''
    replace_by_space = re.compile('[\n/(){}\[\]\|@,;]')
    STOPWORDS = set(stopwords.words('english'))
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z !]')

    s = s.lower()
    s = replace_by_space.sub(' ',s)
    s = BAD_SYMBOLS_RE.sub('', s)
    s = ' '.join(word for word in s.split() if word not in STOPWORDS) # delete stopwors from text
    return s

def clean_df(df):
    '''
    clean text and add some features to it:
     "number of words" column, "sentiment column"

    '''
    # print(df["Text"])

    df = df[["Text"]]
    df["Text"] = df["Text"].apply(clean)
    df["word_count"] = df["Text"].apply(lambda s: sum(Counter(s.split()).values()))
    

    sid = SentimentIntensityAnalyzer()

    df["Sentiments"] = df["Text"].apply(lambda s: sid.polarity_scores(s))
    df = pd.concat([df.drop(['Sentiments'], axis=1), df['Sentiments'].apply(pd.Series)], axis=1)

    return df


##### make predictions

def predict(df):

    df_clean = clean_df(df)

    col_trans = load("./column_trans.joblib")
    model = load("./model.joblib")

    x_transformed = col_trans.transform(df_clean)
    y_pred = model.predict(x_transformed)

    y_pred_df = pd.DataFrame({'Compliment': y_pred[:,0], 'Complaint': y_pred[:,1], 'Suggestion': y_pred[:,2] })
    y_pred_df = pd.concat([df[["Text"]].reset_index(drop=True),y_pred_df.applymap(lambda n:int(n))], axis=1)
    
    return y_pred_df

def highlight_pred(pr):
    '''
    highlight the prediction cells
    '''
    if pr==1:
        return 'background-color: red'
    else:
        return ""

