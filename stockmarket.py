import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.dates as mdates
# Load dataset
df = pd.read_excel('Book3.xlsx')
df.dropna(inplace=True)
df.rename(columns=lambda x: x.strip(), inplace=True)
# clear the instrument names
df['TRADING CODE']=df['TRADING CODE'].str.strip()
da=df[(df['TRADING CODE']=='GP')].sort_values('DATE', ascending=False).reset_index(drop=True)
da=da[da['TRADE']!=0]

my_year_month_fmt = mdates.DateFormatter('%m/%y')
data_1 = pd.read_excel('Book3.xlsx',index_col=[0,1,2,3,4])
data_1.head(10)

short_rolling = data_1.rolling(window=20).mean()
short_rolling.head(20)
# Calculating the long-window simple moving average
long_rolling = data_1.rolling(window=100).mean()
long_rolling.tail()

da['HIGH-LOW']=da['HIGH']-da['LOW']
da['CLOSEP-OPENP']=da['CLOSEP']-da['OPENP']


da['CLOSEP']=da['CLOSEP'].astype('float64')
# Define features and target (assuming the last column is the target)

 
da['Y']=da['CLOSEP'][1:]/da['CLOSEP'][:-1].values-1
da.loc[da['Y']>0, 'DEX']=1
da.loc[da['Y']<=0, 'DEX']=-1

 
da['DEX']=da['DEX'].shift(-1)
da.head()
da.tail()
da=da.dropna()

y=da['DEX']
X=da[['CLOSEP-OPENP', 'HIGH-LOW']]
pc=.90
j=int(pc*len(da['DEX']))
 
X_train=X[:j]
X_test=X[j:]

y_train=y[:j]
y_test=y[j:]


# Load saved model for prediction
stockmarket_model = pickle.load(open(r'C:\Users\AKHILA\Desktop\Stock Market Prediction\stock market.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Deep Learning for Indian Stock Market Prediction: A Study of Nifty and Sensex Trends',
        ['Upload Dataset','Train Random Forest', 'Train Support Vector Machine', 'Train Logistic Regression', 'Stock Market Prediction', 'About'],
        default_index=0
    )

if selected == 'Upload Dataset':
    st.title("Upload Dataset")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df.dropna(inplace=True)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        
        st.write("### First 5 rows of the dataset:")
        st.dataframe(df.head())  # Display first 5 rows
if selected == 'Train Random Forest':
    st.title('Train Random Forest Model')

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    st.write(f'Random Forest Model Accuracy: {accuracy_rf:.2f}')

# Train Support Vector Machine
if selected == 'Train Support Vector Machine':
    st.title('Train Support Vector Machine')

    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    y_pred_svm = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)

    st.write(f'SVM Model Accuracy: {accuracy_svm:.2f}')

# Train Logistic Regression
if selected == 'Train Logistic Regression':
    st.title('Train Logistic Regression')

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)

    st.write(f'Logistic Regression Model Accuracy: {accuracy_lr:.2f}')

# Stock Market Price Prediction
if selected == 'Stock Market Prediction':
    st.title('Stock Market Prediction')
    st.image('download 1.jpg',width=100)

    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        closep_openp = st.text_input('CLOSEP-OPENP')
    
    with col2:
        high_low_input = st.text_input('HIGH-LOW')

    # Prediction button
    if st.button('Stock Market Price Prediction'):
        try:
            prediction = stockmarket_model.predict([[float(closep_openp), float(high_low_input)]])
            st.success(f'The predicted stock market price is: {prediction[0]}')
        except ValueError:
            st.error("Please enter valid numerical values.")
if selected == 'About':
    st.title("About the Project")

    st.write("""
    ### Project Overview
    This project explores **Deep Learning for Indian Stock Market Prediction**, focusing on analyzing 
    trends in **Nifty and Sensex**. It utilizes **machine learning models** such as:
    
    - **Random Forest Classifier**
    - **Support Vector Machine (SVM)**
    - **Logistic Regression**

    The goal is to build predictive models that analyze stock price movements based on historical data.

    ### Key Features:
    - **Data Preprocessing & Cleaning**
    - **Training Machine Learning Models**
    - **Stock Market Price Prediction**
    - **User-Friendly Web Interface using Streamlit**
    
    ### Technologies Used:
    - Python (Pandas, Scikit-learn, Streamlit)
    - Machine Learning Algorithms
    - Pickle for Model Serialization

    This project aims to assist investors in making informed decisions based on historical trends.
    """)
    
    st.image("Capture.jpg")  # Replace with an actual image if available