import numpy as np
import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


# loading the saved model
loaded_model = pickle.load(open(r"C:\Users\Priyanshu Panda\Desktop\Python Files\Fake News\trained_model.sav", 'rb'))
# converting the textual data to numerical data
vectorizer = pickle.load(open(r"C:\Users\Priyanshu Panda\Desktop\Python Files\Fake News\vectorizer.sav", 'rb'))
ps = PorterStemmer()


# creating a function for Prediction
def fakenews_prediction(text):

  review = re.sub('[^a-zA-Z]', ' ', text)
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  review_vect = vectorizer.transform([review]).toarray()
  prediction = loaded_model.predict(review_vect)
  print(prediction)
  if (prediction[0] == 0):
    return 'The news is real'
  else:
    return 'The news is fake'
  
 

def main():
    # giving a title
    st.title('Fake News Prediction Web App')
    # getting the input data from the user
    title = st.text_input('News Title')
    author = st.text_input('Author of the news')
    text = st.text_input('News Text')
    x = title+author+text
    # if (len(x)==0):
    #  st.success("Please Enter Data")
    # code for Prediction
    predict =""
    if st.button('Fake News Result'):
      if (len(x)==0):
        predict = "Enter Data"
      else:
        predict = fakenews_prediction(x)
    st.success(predict)

    st.text('Made By Priyanshu Akash and Sarin')

if __name__ == '__main__':
    main()