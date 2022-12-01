"""
Description
This is a Natural Language Processing(NLP) Based App useful for basic NLP concepts such as follows;

+ Document/Text Summarization using Gensim/Sumy
This is built with Streamlit Framework, an awesome framework for building ML and NLP tools.
Purpose
To perform basic and useful NLP task with Streamlit,Spacy,Textblob and Gensim/Sumy
"""
# Imports
import streamlit as st
import os
import numpy as np
import pandas as pd
import requests

def main():
    """ Find out Code Creator with Code Recognizer  """

    # Title
    st.title("Code Recognizer Φ")
	
    #3. User Input
    user_input = st.text_area("Enter the code below.")

    result = st.button('Find out creator')
    user_input

    #Calling API
    url = ('https://xref-tf3z57rlzq-ew.a.run.app')

    json={'text':user_input}
    response = requests.post(url, json=json)

    response.text

    direction = st.radio('Learn more about models used in app code', ('Naive Bayes', 'RNN', 'CNN'))
    if direction == 'Code':
        st.write('')
    elif direction == 'Naive Bayes':
        st.markdown ('Naive Bayes')    
        st.write('''
 is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.
It is mainly used in text classification that includes a high-dimensional training dataset.
Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.
It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.
Some popular examples of Naïve Bayes Algorithm are spam filtration, Sentimental analysis, and classifying articles.'

Application:
''')
    elif direction == 'RNN':
        st.write('''RNN:

Description:
"Recurrent Networks are one such kind of artificial neural network that are mainly intended to identify patterns in data sequences, such as text, genomes, handwriting, the spoken word, numerical times series data emanating from sensors, stock markets, and government agencies".

Application:
''')
    elif direction == 'CNN':
        st.write('''CNN:

Description:
"Recurrent Networks are one such kind of artificial neural network that are mainly intended to identify patterns in data sequences, such as text, genomes, handwriting, the spoken word, numerical times series data emanating from sensors, stock markets, and government agencies".

Application:
''')
    else:
        st.write('◀️')
    
    st.sidebar.subheader("Code Recognizer Φ")
    st.sidebar.info(
     """  About: 
     
        What:
        _ Criado no ano de 2022 no batch 1011 do 
        Le Wagon Brasil o  “Code Recognizer" é um 
        aplicativo que usa inteligência artificial 
        para identificar o autor de um código de 
        programação. 
        _ Multilingual model: 
        3 languages: java, C++, python
        
        Why: 
        _ Identify malicious software
        _ Solve intellectual property disputes
        _ Tell who's Jr from who's Sr 
        
        How: 
        _ Google's Code Jam archives as a dataset 
        - Famous coding competition hosted by Google. 
        _ Thousands of solutions to comparable problems. 
        _ Data from 2008 to 2020.
        _ Tested Models: Naive Bayes, RNN e CNN
        _ Features Selection : > 9 features
        _ Estudo de Referência: Code authorship 
          identification using convolutional neural
          networks University, Incheon, South Korea)
        
        How: 
        _ Google's Code Jam archives (*as a dataset) 
        _ Famous coding competition hosted by Google. 
        _ Thousands of solutions to comparable problems.
        _ Data from 2008 to 2020.
        _ Modelos Testados: Naive Bayes, RNN e CNN
        _ Features Selection : > 9 features
        _ Estudo de Referência: Code authorship identification using convolutional neural networks University, Incheon, South Korea
""")     
        
    st.sidebar.text("""
        Created by
        
        https://www.linkedin.com/in/joaopaulobaccarin/
        https://www.linkedin.com/in/rafaelincao
        https://www.linkedin.com/in/timcerta/
        https://www.linkedin.com/in/thomasgrumberg
        """)
#url = "https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py"
#st.write("check out this [link](%s)" % url)

#st.markdown("check out this [link](%s)" % url)

if __name__ == '__main__':
	main()