import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

import psycopg2

#initialize connection
@st.experimental_singleton
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn=init_connection()
print('connection established')

#perform query
@st.experimental_memo(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()
        count=cur.rowcount
        print(count,'record inserted')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf3 = pickle.load(open('vectorizer_2.pkl','rb'))
model3 = pickle.load(open('model_2.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms3 = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf3.transform([transformed_sms3])
    # 3. predict
    result3 = model3.predict(vector_input)[0]
    # 4. Display
    if result3 == 1:
        st.header("Spam")
        query=f'''
        insert into "Dataset" (target,text)
        values('spam','{input_sms}');
        '''
        run_query(query)
        print('query executed')
    else:
        st.header("Not Spam")
        query = f'''
                insert into "Dataset" (target,text)
                values('ham','{input_sms}');
                '''
        run_query(query)
        print('query executed')