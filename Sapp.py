import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st 
import matplotlib.pyplot as plt
model=load_model('STP.keras')

st.header("Stock Price Prediction")
stock=st.text_input("Enter Stock Symbol",'GOOG')
start='2010-01-01'
end='2024-01-01'
data=yf.download(stock,start,end)

st.subheader('Stock Data')
st.write(data)

data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

pas_100_days=data_train.tail(100)
data_test=pd.concat([pas_100_days,data_test],ignore_index=True)
data_test_scale=scaler.fit_transform(data_test)

    


st.subheader('Price VS Moving AVERAGE 100')
ma_100_days=data.Close.rolling(100).mean()
fig2=plt.figure(figsize=(8,6))

plt.plot(ma_100_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price VS Moving AVERAGE 100 VS Moving AVERAGE 200')
ma_200_days=data.Close.rolling(200).mean()
fig3=plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig3)


a=[]
b=[]

for i in range(100,data_test_scale.shape[0]):
    a.append(data_test_scale[i-100:i])
    b.append(data_test_scale[i,0])
a,b=np.array(a),np.array(b)    

predict=model.predict(a)

scale= 1/scaler.scale_

predict=predict*scale
b=b*scale


st.subheader('Orginal VS Predicted Price')
fig4=plt.figure(figsize=(8,6))
plt.plot(predict,'r',label='Orginal Price')
plt.plot(b,'g',label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()
st.pyplot(fig4)

def forecast_next_50_days(model, data, scaler):
    # Use the last 50 days from the training data for forecasting
    last_50_days = data[-100:].reshape(1, -1)
    
    # Scale the data
    last_50_days_scaled = scaler.transform(last_50_days.reshape(-1, 1))
    
    # Reshape the data for the LSTM model
    X_input = last_50_days_scaled.reshape((1, 100, 1))
    
    # Initialize a list to store the predicted stock prices
    predicted_prices = []
    
    # Predict the next 50 days
    for _ in range(50):
        predicted_price = model.predict(X_input)
        
        # Rescale the prediction back to the original scale (inverse scaling)
        predicted_price_rescaled = scaler.inverse_transform(predicted_price)
        
        # Append the predicted price to the list
        predicted_prices.append(predicted_price_rescaled[0][0])
        
        # Update the input sequence by adding the predicted price to the last sequence
        X_input = np.append(X_input[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
    
    return predicted_prices

# Forecast the next 50 days
predicted_prices = forecast_next_50_days(model, data['Close'].values, scaler)

# Generate a date range for the next 50 days
forecast_date_range = pd.date_range(data.index[-1], periods=51, freq='D')[1:]

# Plotting the forecast for the next 50 days
st.subheader('Stock Price Forecast for the Next 100 Days')
fig5 = plt.figure(figsize=(10, 6))
plt.plot(forecast_date_range, predicted_prices, color='blue', label='Forecasted Price')
plt.title(f'{stock} 50-Day Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig5)
