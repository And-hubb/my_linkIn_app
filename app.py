
import pandas as pd
import os
import numpy as np
import altair as alt
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Q1
s = pd.read_csv("social_media_usage.csv")
#s.shape
#(1502, 89)
#Q2
def clean_sm(x):
    x = np.where(x == 1, 1,0)
    return x
 
#df_toy = pd.DataFrame({
#    "job": [1,0,4],
#    "work": [0,0,1]})
#df_toy

#clean_sm(df_toy)
#array([[1, 0],
    #    [0, 0],
    #    [0, 1]])
#Q3
ss_all = pd.DataFrame({  #gathering all data before filtering
    "sm_li": clean_sm(s["web1h"]), #running clean_sm on "web1h" indicates if Linkedin User
    "income": (s["income"]),
    "education": (s["educ2"]),
    "parent": (s["par"]),
    "married": (s["marital"]),
    "female": (s["gender"]),
    "age": (s["age"])
})

ss_income = ss_all[ss_all["income"] < 10] #removing/dropping missing values from "income"
ss_edu = ss_income[ss_income["education"] <= 8] #removing missing values from "education"
ss_parent = ss_edu[ss_edu["parent"] <= 2] #removing missing values from "parent"
ss_married = ss_parent[ss_parent["married"] <= 6] #removing missing values from "married"
ss_female = ss_married[ss_married["female"] <= 3] #removing missing values from "female"
ss_age = ss_female[ss_female["age"] < 98] #removing missing values from "age"
ss_filtered = ss_age #final filterd data removing missing values

def clean_female(y): #assignment female if female to make binary
    y = np.where(y == 2, 1,0)
    return y

ss = pd.DataFrame({  #rest of cleaning
    "sm_li": (ss_filtered["sm_li"]), #binary
    "income": (ss_filtered["income"]),
    "education": (ss_filtered["education"]),
    "parent": clean_sm(ss_filtered["parent"]), #binary
    "married": clean_sm(ss_filtered["married"]), #binary
    "female": clean_female(ss_filtered["female"]), #binary
    "age": (ss_filtered["age"])
})


# alt.Chart(ss.groupby(["education", "income"], as_index=False)["sm_li"].mean()).\
# mark_circle().\
# encode(x="education",
#       y="sm_li",
#       color="income:N")
#seems the more education the more likeley they are to be a linkedIn user
#pd.crosstab(ss["education"], columns ="count", normalize=True)
# col_0	count
# education	
# 1	0.0112
# 2	0.0304
# 3	0.1984
# 4	0.1648
# 5	0.1080
# 6	0.2656
# 7	0.0272
# 8	0.1944
# Q4
Y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]
#Q5
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    stratify=Y,       
                                                    test_size=0.2,    
                                                    random_state=987) 
# X_train is used to predict "sm_li" which is if they will be a linkedIn user or not, and it has 80% of the "ss" data 
# X_test is used to run our prediction formula on this data so we can see how accurate our model will be. This has 20% 
    #of the "ss" data.
# y_train has the "sm_li" which we will use to see what "X_train" features predict when "sm_li" is 1. 
    #This contains 80% of the matching data that "X_train" has. 
# y_test has 20% of the data which has the solution to "X_test" data. This is how we will know how accurate our model will be.
#Q6
lr = LogisticRegression()
lr.fit(X_train, y_train)
LogisticRegression()
#Q7
y_pred = lr.predict(X_test)
#print(classification_report(y_test, y_pred))
#print(classification_report(y_test, y_pred)) #accuracy is 70%
#               precision    recall  f1-score   support

#            0       0.74      0.85      0.79       166
#            1       0.58      0.40      0.48        84

#     accuracy                           0.70       250
#    macro avg       0.66      0.63      0.63       250
# weighted avg       0.68      0.70      0.68       250

# pd.DataFrame(confusion_matrix(y_test, y_pred),
#             columns=["Predicted negative", "Predicted positive"],
#             index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")
#confusion_matrix(y_test, y_pred)
# array([[141,  25],
#        [ 50,  34]], dtype=int64)
#the model accuratley predicted 141 negative that were negative. 
#the model accurately predicted 34 positive that were positive.
#the model incorrectly predicted 25 as positive that were actually negative
#Lastly, the model predicted 50 negative that were actually positive.
#The model was 70% accurate
#Q8
# col_1 = ["True Negative","Fale Negative"]
# col_2 = ["Fale Positive","True Positive"]
# df_confusion_matrix = pd.DataFrame((col_1, col_2), 
#     columns=["Predicted negative", "Predicted positive"],
#     index=["Actual negative","Actual positive"])
# df_confusion_matrix
#Predicted negative	Predicted positive
#Actual negative	True Negative	Fale Negative
#Actual positive	Fale Positive	True Positive
#pd.DataFrame(confusion_matrix(y_test, y_pred),
#            columns=["Predicted negative", "Predicted positive"],
#            index=["Actual negative","Actual positive"])
#Predicted negative	Predicted positive
#Actual negative	141	25
#Actual positive	50	34
#Q9
#precision = 34/(34+25) #= precision
#precision #= precision is important when you want to reduce the chance of incorrectly predicting positive cases, like Covid Testing.
#0.576271186440678
# important if you want to minimize the occurance of missing positive instances, like Fraud.
#recall = 34/(34+50)
#recall #= recall is important if you want to minimize the occurance of missing positive instances, like Fraud.
#0.40476190476190477
#, like predicting the outcome of a game
#f1_score = 2*((precision*recall)/(precision+recall))
#f1_score #= F1 Score is a good middle ground if neither of the above are particullarly important, like predicting the outcome of a game.
#0.4755244755244756
 #they match what I calculated by hand
#print(classification_report(y_test, y_pred)) #they match what I calculated by hand
#               precision    recall  f1-score   support

#            0       0.74      0.85      0.79       166
#            1       0.58      0.40      0.48        84

#     accuracy                           0.70       250
#    macro avg       0.66      0.63      0.63       250
# weighted avg       0.68      0.70      0.68       250

#Q10
person_42 = [8,7,0,1,1,42]
probs_42 = lr.predict_proba([person_42])
predict_42 = lr.predict([person_42])
person_82 = [8,7,0,1,1,82]
probs_82 = lr.predict_proba([person_82])
predict_82 = lr.predict([person_82])

################################################################
####################### for APPPPPPPP ##########################
#test connecting to app
#st.markdown(f"#Probability that 42 year old sample individual uses LinkedIn is: {probs_42[0][1]}")
#test connecting to app and log model
#st.markdown(f"##Probability that 82 year old sample individual uses LinkedIn is: {probs_82[0][1]}")
#st.markdown(predict_42)
#st.markdown(f"#Probability that 42 year old sample individual uses LinkedIn is: {predict_82[0][1]}")

#test connecting to app and log model
# if predict_42 == 1:
#     st.markdown("They will be a LinkedIn User")
# else:
#     st.markdown("They are not a LinkedIn User")

st.markdown("Select the information below that matches the individual you would like to test and predict if they would be classified as a LinkedIn User:")
st.markdown(" ")

########## SLIDERS FOR INPUTS

# num1 = st.slider(label="Enter a number for income", 
#           min_value=1,
#           max_value=9,
#           value=4)

num1 = st.selectbox("Select Income Level", 
             options = ["Less than $10,000",
                        "$10,000 to under $20,000",
                        "$20,000 to under $30,000",
                        "$30,000 to under $40,000",
                        "$40,000 to under $50,000",
                        "$50,000 to under $75,000",
                        "$75,000 to under $100,000",
                        "$100,000 to under $150,000",
                        "$150,000 or more"])

# st.write(f"Income Level (pre-conversion): {num1}")  ## not needed

# st.write("**Convert Selection to Numeric Value**")   ## not needed

if num1 == "Less than $10,000":
    num1 = 1
elif num1 == "$10,000 to under $20,000":
    num1 = 2
elif num1 == "20,000 to under $30,000":
    num1 = 3
elif num1 == "$30,000 to under $40,000":
    num1 = 4
elif num1 == "$40,000 to under $50,000":
    num1 = 5
elif num1 == "$50,000 to under $75,000":
    num1 = 6
elif num1 == "$75,000 to under $100,000":
    num1 = 7
elif num1 == "$100,000 to under $150,000":
    num1 = 8
else:
    num1 = 9

# num2 = st.slider(label="Enter a number for education",
#           min_value=1,
#           max_value=8,
#           value=4)

num2 = st.selectbox("Select Education Level", 
             options = ["Less than high school (Grades 1-8 or no formal schooling)",
                        "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                        "$High school graduate (Grade 12 with diploma or GED certificate)",
                        "Some college, no degree (includes some community college)",
                        "Two-year associate degree from a college or university",
                        "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)",
                        "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                        "Postgraduate or professional degree, including master's, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"])

if num2 == "Less than high school (Grades 1-8 or no formal schooling)":
    num2 = 1
elif num2 == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    num2 = 2
elif num2 == "High school graduate (Grade 12 with diploma or GED certificate)":
    num2 = 3
elif num2 == "Some college, no degree (includes some community college)":
    num2 = 4
elif num2 == "Two-year associate degree from a college or university":
    num2 = 5
elif num2 == "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)":
    num2 = 6
elif num2 == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    num2 = 7
else:
    num2 = 8

# num3 = st.slider(label="Enter a number for parent",  #### needs to be yes/no check/slider
#           min_value=0,
#           max_value=1,
#           value=0)

num3 = st.selectbox("Are they a parent of a child under 18 living in their home?", 
             options = ["No, they are not a parent",
                        "Yes, they are a parent"])

if num3 == "No, they are not a parent":
    num3 = 0
else:
    num3 = 1

# num4 = st.slider(label="Enter a number for married",   #### needs to be yes/no check/slider
#           min_value=0,
#           max_value=1,
#           value=0)

num4 = st.selectbox("Are they married?", 
             options = ["No, they are not married",
                        "Yes, they are married"])

if num4 == "No, they are not married":
    num4 = 0
else:
    num4 = 1


# num5 = st.slider(label="Enter a number for female",   #### needs to be yes/no check/slider
#           min_value=0,
#           max_value=1,
#           value=0)

num5 = st.selectbox("Are they Male or Female?", 
             options = ["They are Male",
                        "They are Female"])

if num5 == "They are Male":
    num5 = 0
else:
    num5 = 1

num6 = st.slider(label="Enter a number for their age", 
          min_value=1,
          max_value=97,
          value=50)

person_streamlit = [num1,num2,num3,num4,num5,num6]
prob_streamlit = lr.predict_proba([person_streamlit])
predict_streamlit = lr.predict([person_streamlit])


#put % gauge here if you get to work


#line between entries and outputs
st.markdown(" ")
st.markdown("Based on the information you previded it is predicted that:")

#text for % likely
st.markdown(f"Probability that the individual uses LinkedIn is: {prob_streamlit[0][1]}")


#text for they will or will not be a linkedin user
if predict_streamlit == 1:
    st.markdown("They would be classified as a LinkedIn user")
else:
    st.markdown("They are not going to be classified as LinkedIn user")

############## Gauge for % likely  #################################################
#### Create sentiment gauge
#wk3 59:41 app.py code
# st.markdown("# Sentiment Analysis Application!!")

# #### Create text input box and save incoming text to variable called text
# text = st.text_input("Enter text:", value = "Enter text here")

# #### TextBlob to analyze input text
# score = TextBlob(text).sentiment.polarity

# #### Create label (called sent) from TextBlob polarity score to use in summary below
# if score > .15:
#     label = "Positive"
# elif score < -.15:
#     label = "Negative"
# else:
#     label = "Neutral"
    
# ##### Show results

# #### Print sentiment score, label, and language
# st.markdown(f"Sentiment label: **{label}**")

# fig = go.Figure(go.Indicator(
#     mode = "gauge+number",
#     value = score,
#     title = {'text': f"Sentiment: {label}"},
#     gauge = {"axis": {"range": [-1, 1]},
#             "steps": [
#                 {"range": [-1, -.15], "color":"red"},
#                 {"range": [-.15, .15], "color":"gray"},
#                 {"range": [.15, 1], "color":"lightgreen"}
#             ],
#             "bar":{"color":"yellow"}}
# ))


# st.plotly_chart(fig)
