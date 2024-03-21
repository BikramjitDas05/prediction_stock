import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
########################################################################
start = '2000-01-01'
end = '2019-12-31'

st.title("Stock Price Prediction")
user_input = st.text_input("Enter Stock Ticker")

if user_input:
    try:
        df = yf.download(user_input, start=start, end=end)
        print(df.head())

#Describing Data

        st.subheader('Data from 2000 - 2019')
        st.write(df.describe())

#Visualization
        st.subheader('Closing Price vs Time chart')
        fig = plt.figure(figsize = (12,6))
        plt.plot(df.Close)
        st.pyplot(fig)

#100 days Moving Average
        st.subheader('Closing Price vs Time chart with 100 Moving Average')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize = (12,6))
        plt.plot(ma100)
        plt.plot(df.Close)
        st.pyplot(fig)

#100 & 200  days Moving Average
        st.subheader('Closing Price vs Time chart with 100 Moving Average & 200 Moving Average')
        ma100  =df.Close.rolling(100).mean()
        ma200  =df.Close.rolling(200).mean()
        fig = plt.figure(figsize = (12,6))
        plt.plot(ma100,'r')
        plt.plot(ma200,'g')
        plt.plot(df.Close,'b')
        st.pyplot(fig)

#Spliting the data into train and test
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

        print(data_training.shape)
        print(data_testing.shape)


        scaler = MinMaxScaler(feature_range=(0,1))

        data_training_array = scaler.fit_transform(data_training)




#Load My Model
        model = load_model('keras_model.h5')

################################################
#Fit my model into testing
        past_100_days = data_training.tail(100)
        final_df = past_100_days.append(data_testing, ignore_index = True)
        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i,0])

        x_test, y_test = np.array(x_test), np.array(y_test)
#make prediction
        y_predicted = model.predict(x_test)

        scaler = scaler.scale_

        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

#Visualization
        st.subheader('Predictions vs Original')
        fig2 = plt.figure(figsize =(12,6))
        plt.plot(y_test, 'b',label = "Original Price")
        plt.plot(y_predicted, 'r',label = "Predicted Price")

        plt.xlabel('Time')
        plt.ylabel('Price')
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error fetching or plotting data: {e}")




