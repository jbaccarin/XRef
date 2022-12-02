#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.express as px

st.set_page_config(page_title="Xref - Code Authorship Attribution", page_icon="⚙️", initial_sidebar_state="expanded")

##########################################
##  Load and Prep Data                  ##
##########################################

##########################################
##  Style and Formatting                ##
##########################################

# # CSS for tables

# hide_table_row_index = """
#             <style>
#             thead tr th:first-child {display:none}
#             tbody th {display:none}
#             </style>   """

# center_heading_text = """
#     <style>
#         .col_heading   {text-align: center !important}
#     </style>          """

# center_row_text = """
#     <style>
#         td  {text-align: center !important}
#     </style>      """

# # Inject CSS with Markdown

# st.markdown(hide_table_row_index, unsafe_allow_html=True)
# st.markdown(center_heading_text, unsafe_allow_html=True)
# st.markdown(center_row_text, unsafe_allow_html=True)

# # More Table Styling

# def color_surplusvalue(val):
#     if str(val) == '0':
#         color = 'azure'
#     elif str(val)[0] == '-':
#         color = 'lightpink'
#     else:
#         color = 'lightgreen'
#     return 'background-color: %s' % color

# heading_properties = [('font-size', '16px'),('text-align', 'center'),
#                       ('color', 'black'),  ('font-weight', 'bold'),
#                       ('background', 'mediumturquoise'),('border', '1.2px solid')]

# cell_properties = [('font-size', '16px'),('text-align', 'center')]

# dfstyle = [{"selector": "th", "props": heading_properties},
#                {"selector": "td", "props": cell_properties}]

# Expander Styling

st.markdown(
    """
<style>
.streamlit-expanderHeader {
 #   font-weight: bold;
    background: aliceblue;
    font-size: 18px;
}
</style>
""",
    unsafe_allow_html=True,
)



##########################################
##  Title, Tabs, and Sidebar            ##
##########################################

st.title("XRef_")
st.markdown('''##### <span style="color:gray">Code Authorship Attribution</span>
            ''', unsafe_allow_html=True)

tab_nn, tab_cnn, tab_svc, tab_faq = st.tabs(["NN Model", "CNN Model", "SVC Model", "FAQ"])

col1, col2, col3 = st.sidebar.columns([1,1,1])
with col1:
    st.write("")
with col2:
    st.image('figures/x.png',  use_column_width=True)
with col3:
    st.write("")

st.sidebar.markdown(" # About")
st.sidebar.markdown("This aplication uses deep learning models to predict the true author of a sample of code based on its stylometry.")
st.sidebar.markdown("Such models were trained with source code extracted from Google Code Jam - a very famous coding competition.")
st.sidebar.markdown(" # Created by")
st.sidebar.markdown("João Baccarin [LinkedIn](https://www.linkedin.com/in/joaopaulobaccarin/)")
st.sidebar.markdown("Tim Certa [LinkedIn](https://www.linkedin.com/in/timcerta/)")
st.sidebar.markdown("Thomas Grumberg [LinkedIn](https://www.linkedin.com/in/thomasgrumberg)")
st.sidebar.markdown("Rafael Incao [lLinkedIn](https://www.linkedin.com/in/rafaelincao)")
st.sidebar.info("ℹ️ Read more about our project and check the code at: [Github](https://github.com/jbaccarin/xref).")

##########################################
## NN Tab                               ##
##########################################

with tab_nn:
    st.markdown('''**Model description:**
    Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.

    For this model we used:
        - A TF-IDF vectorizer (with top 5000 features, including non-word characters)
        - 2 dense layers 1000 nodes deep, with ReLu activation functions
        - 2 subsequent drop-out layers with 20% keep rate
        - Final dense layer with Softmax activation
        - Adam Optimizer''')
    st.markdown('''Accuracy on test data: **89%**''')
    st.markdown("---")

##########################################
## CNN Tab                              ##
##########################################

with tab_cnn:
    st.markdown('''**Model description:**
    Convolutional neural network (CNN), a class of artificial neural networks that has become dominant in various computer vision tasks, is attracting interest across a variety of domains, including radiology. CNN is designed to automatically and adaptively learn spatial hierarchies of features through backpropagation by using multiple building blocks, such as convolution layers, pooling layers, and fully connected layers.

    For this model we used:
        - A TF-IDF vectorizer (with top 5000 features, including non-word characters)
        - 3 1D convolutional layers with kernels of size [3,4,5] and ReLu activation
        - 3 subsequent drop-out layers with 50% keep rate
        - 1 Flatten layer
        - Final dense layer with Softmax activation
        - Adam Optimizer''')

    st.markdown('''Accuracy on test data: **92%**''')
    st.markdown("---")

    user_input = st.text_area("Enter the code below.", )

    #Calling API
    url = 'https://xref-app-tf3z57rlzq-ew.a.run.app'
    predict_url = "https://xref-app-tf3z57rlzq-ew.a.run.app/predict_author_with_NN?code=https%3A%2F%2Ftaxi-instance-tf3z57rlzq-ew.a.run.app"


    # Find out the author for the given piece of code
    if st.button('Find out code author'):
        with st.spinner('Wait for it...'):
            st.write('Please wait, the author is being identified...')
        params = dict(
                code=[user_input]
            )

        print('Author is being identified')
        st.write(" ")
        st.write(" ")

        response = requests.get(predict_url, params).json()
        res = response["author"]
        proba = response["probabilities"]
        st.success('Done!')
        # Return result
        st.subheader(f"The predicted author is {res}!")


        st.write(" ")
        st.write(" ")

        # Calculate and display probabilities
        #st.subheader(f"Top 5 probabilities among all programmers:")
        #st.bar_chart(data = data_)
        data_ = pd.DataFrame.from_dict(proba, orient='index')
        data_.rename({0:'Probability'},inplace=True,axis=1)
        data_['Probability'] = (data_['Probability']*100).astype(int)
        data_ = data_.reset_index()
        data_.rename({"index":'Author'},inplace=True,axis=1)
        fig=px.bar(data_,x='Author',y='Probability', orientation='v', title='<span style="font-size: 30px;">Top 5 probabilities among all programmers</span>')
        fig.update_layout(height=400, width = 770, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        st.plotly_chart(fig)


        st.write(" ")
        st.write(" ")

        # Calculate and show tfidf terms
        #st.subheader("Most important terms identified by tf-idf vectorizer:")
        top_terms_dict = response["top_terms"]
        topterms = pd.DataFrame.from_dict(top_terms_dict, orient='index')
        topterms.rename({0:'Tfidf'},inplace=True,axis=1)
        topterms = topterms.reset_index()
        topterms.rename({'index':'Term'},inplace=True,axis=1)
        #st.table(data = topterms)
        fig=px.bar(topterms,x='Term',y='Tfidf', orientation='v', title='<span style="font-size: 30px;">Most important terms identified by tf-idf vectorizer</span>')
        fig.update_layout(height=400, width = 770, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        st.plotly_chart(fig)

##########################################
## SVC Tab                              ##
##########################################

with tab_svc:
        st.markdown('''**Model description:**
        A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems. After giving an SVM model sets of labeled training data for each category, they’re able to categorize new text.''')
        st.markdown('''Compared to newer algorithms like neural networks, they have two main advantages: higher speed and better performance with a limited number of samples (in the thousands). This makes the algorithm very suitable for text classification problems, where it’s common to have access to a dataset of at most a couple of thousands of tagged samples.


        For this model we used:
        - A TF-IDF vectorizer (with top 5000 features, including non-word characters)
        - LinearSVC - uses a linear kernel (scales better with larger samples)
        - Bayesian Optimization for hyper-parameter tuning:
            - Best n-gram sizes (2,3)
        - Suggested by Ellen, based on some of her previous projects''')
        st.markdown('''Accuracy on test data: **85%**''')
        st.markdown("---")






##########################################
## FAQ Tab                              ##
##########################################

with tab_faq:
    st.markdown(" ### Frequently Asked Questions 🔎 ")

    ##########
    expand_faq1 = st.expander("Why use TF-IDF instead of other vectorizers? ")
    with expand_faq1:

        st.write('''It's well known among basketball lovers that an NBA player's salary often doesn't reflect his true market value.  Most players play on guaranteed multi-year contracts, over the course of which their performance may improve or deteriorate, often dramatically;  yet their salaries remain locked-in.  A useful method for determining any player's fair market value could prove worthwhile both for team executives (to exploit inefficiencies and assemble a competitive team on a budget) and for regular fans (to quantify how overpaid or underpaid their favorite players really are).

What, then, is an NBA player's true market value?  To answer this most pressing of society's questions, I came up with a simple but surprisingly powerful approach. Every year, out of several hundred NBA players, about 150 become free agents and sign new contracts.  My idea here was to focus exclusively on this subset of players in line for a new contract, to see if I could train a model to predict their new salaries from their previous year's stats.
By looking at all free agents over the course of several years, we can use machine learning techniques to uncover a mapping from a player's stats in the last year of his contract to his resulting new salary the following year.  The key insight is that once we have this mapping in hand, we can then retroactively apply it to ALL players (not just free agents).  In other words, we can answer the question: "*If Player X were a free agent this year*, what kind of new salary would he command based on his current stats?"  It is important to note that my model is not making a normative judgment ("this is a good player/ this is a bad player").  Rather, it is saying: "in the recent past, players in line for a new contract— with stats like those of Player X this year— could expect to get Salary Y."  ''', unsafe_allow_html=True)

    ##########
    expand_faq2 = st.expander("What machine learning model did we use?")
    with expand_faq2:

        st.write('''I tried various regression, classification, and hybrid approaches and found that using  a Random Forest Classifier as my predictive model gave accurate and meaningful results. A Random Forest is an ensemble model consisting of thousands of Decision Trees, with each tree constructed from a random bootstrapped sample of players in the training set; each node on each tree is split using a random sample of the feature (input) variables. The values of hyperparameters such as maximum tree depth and  number of features considered at each node were arrived at via grid search optimization.
For my classification target variable, I grouped the free agent next-year salaries into seven buckets: \$0-5M, \$5-10M, \$10-15M, \$15-20M, \$20-25M, \$25-30M, and \$30M+, and chose accuracy as my optimization metric.  Importantly, I made sure to balance these seven classes before model training, to prevent model bias toward the dominant class (after all, over half of all players earn \$0-5M, so a reasonably accurate but utterly useless model could just naively guess this class every time!).''', unsafe_allow_html=True)

    ##########
    expand_faq3 = st.expander("How was the predictive model trained?", expanded=False)
    with expand_faq3:

        st.write('''To train my model, I collected data for all free agents from 2015 to 2020 (the NBA salary cap had a massive spike in a 2015 due to a sudden influx of money from a new TV deal, so it made sense to use that as the cutoff year). For each player, I  used his stats in the final year of his old contract as the feature (input) variables and his new salary the following year as the target (output) variable. I also normalized each salary by that year's salary cap , since teams evaluate salaries as a percentage of the salary cap, rather than by the specific dollar amount.
This gave me 744 total entries (or about 150 free agents per year).  First, I took all the entries from 2020 and siloed them away from my own prying eyes, to use later as a holdout set for testing final model performance.  I then used stratified sampling to split the remaining entries from 2015 to 2019 into a training set (for learning model parameters) and a validation set (for comparing different models and tuning hyperparameters).
After settling on final model hyperparameter values using the validation set, I trained a model on the combined training + validation sets and evaluated its performance using the 2020 holdout set. Finally, I recombined all 744 entries (training + validation + holdout) and used this full dataset to train a final model with the same hyperparameters as above. It is this final model that is used to generate the 2021 market value predictions seen in the web app.''', unsafe_allow_html=True)

    ##########
    expand_faq4 = st.expander("Which dataset does the model use as input variables?")
    with expand_faq4:

        st.write('''After some iteration, I found that using the following set of eight stats as features (input variables) made for a robust and accurate model:
1. <font color=blue>**Points Per Game**</font>
2. <font color=blue>**Minutes Per Game**</font>
3. <font color=blue>**Games Started/Games Played:**</font>  The fraction of a player's games that he started.
4. <font color=blue>**Games Started/Team Games:**</font> The fraction of a team's total games that a player started (normally the denominator is 82, but slightly fewer games were played in the 2019 and 2020 Covid-shortened seasons.)
5. <font color=blue>**Usage Percentage:**</font>  The fraction of team possessions with the player on the court that end in him shooting the ball, turning it over, or getting to the free throw line.
6. <font color=blue>**Offensive Box Plus/Minus (OBPM):**</font> A box score-based metric that estimates a player’s contribution to the team offense while that player is on the court, measured in points above league average per 100 possessions played.
7. <font color=blue>**Value Over Replacement Player (VORP):**</font>  Similar to Offensive Box Plus/Minus above, but also takes into account a player's defensive contributions and scales with playing time and number of games played.
8. <font color=blue>**Win Shares (WS):**</font>  An advanced stat that aims to assign credit for team wins to individual player performance. Win Shares are calculated using player, team and league-wide statistics, with the end result that the sum of player win shares on a given team will be roughly equal to that team’s win total for the season.
It was heartening to see that this set of features included both rate stats (measuring player performance per minute or per possession) and volume stats (taking into account playing time as well), since a truly valuable player should demonstrate good performance on both. For anyone curious about how other features such as age, height, and shooting percentage correlate with market value, check out the feature-target plots in my [modeling notebook](https://github.com/andreilevin/HoopsHero/blob/main/3-model.ipynb).''', unsafe_allow_html=True)

    ##########
    expand_faq5 = st.expander('''So, how good is this model really?''')
    with expand_faq5:

        st.write('''As mentioned previously, I used accuracy as my training optimization metric. In other words, I tried to maximize the percentage of free agents that my model places in the correct next-year salary bucket based on their previous-year stats. When I evaluated the model on the holdout set of 2020 free agents, it produced a very encouraging accuracy of 68%.
However, it would do no good to be 68% accurate if the remaining 32% misclassified entries were all over the map (for example by frequently classifying \$0-5M value players as having a value of \$25-30M, or vice versa).  We can visualize misclassifications with the help of a confusion matrix, shown below for all 156 players in the holdout test set. The rows show the next-year salary buckets as predicted by the model, and the columns show the actual next-year salary buckets. Every player must fall into one of the 49 elements of the 7-by-7 confusion matrix, depending on his predicted and actual salaries: ''', unsafe_allow_html=True)

        st.markdown('''<p style="text-align:center;"><img src="https://raw.githubusercontent.com/andreilevin/HoopsHero/main/figures/confmatrix_test.png"
      title="Confusion Matrix" width="550"/></p>''', unsafe_allow_html=True)

        st.write('''A perfect model with 100% classification accuracy would only have elements on the diagonal of the confusion matrix. We see indeed that 68% of the players in our holdout set lie on the diagonal (note too that most free agents end up making \$0-5M in actual salary). The 32% remaining misclassified players make up the off-diagonal elements. A certain amount of misclassification is to be expected, since our set of 8 features cannot possibly account for the myriad quantifiable and unquantifiable variables that actually determine a player's salary. However, the combination of 68% accuracy with a general lack of extreme off-diagonal entries indicates that the model is pretty reliable, and can be trusted to not embarrass me in job interviews.''', unsafe_allow_html=True)

    ##########
    expand_faq6 = st.expander("How do you calculate authorship probability?")
    with expand_faq6:

        st.write(''' "Surplus value" is my conservative estimate of the difference between a player's market value and his salary.  I calculate it as follows:
* If the player's salary falls within his market value bucket, I define his surplus value as zero
* If the player's salary is higher/lower than his market value, his surplus value is the difference between his salary and the higher/lower end of his market value bucket.
Calculated this way, a player's true absolute surplus value will sometimes be underestimated, but never overestimated.
''', unsafe_allow_html=True)

    ##########
    expand_faq7 = st.expander("What happens if the model isn't sure about a prediction? ")
    with expand_faq7:

        st.write('''This is true— for example, in the 2021-22 season, Stephen Curry had the highest salary in the league at \$45.8M. But recall that the question we are trying to answer is: "what new salary would Player X command if he became a free agent this year?"  By NBA rules, the maximum allowable salary in the first year of a new contract currently ranges from \$28.1M to \$39.3M, depending on how long the player has been in the league (in subsequent years of the contract, the salary can and does increase beyond this range, as we clearly see with Mr. Curry above).
Empirically speaking, if a player is talented enough to be eligible for a salary of \$30M+, teams will generally offer the highest salary available to him, rather than a few million less (the NBA is very much a star-driven league, so teams try to avoid potentially antagonizing star players or their agents).  As far as market value goes, there is thus hardly any difference between a \$30M/year player and a \$40M/year player, and it makes sense to group all such "max players" into a single \$30M+ bucket.  ''', unsafe_allow_html=True)

    ##########
    expand_faq8 = st.expander("Where can I see the code for the model?")
    with expand_faq8:

        st.write('''Glad you asked! 🤓 It's all on our [Github](https://github.com/jbaccarin/xref)''', unsafe_allow_html=True)
