
##visualzing of time series
def plot_df(df,x,y,title="",xlabel="",ylabel=""):
    plt.figure(figsize=(10,10))
    plt.plot(x,y,color="red")
    #plt.show()

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    link = "C:/Users/user/Downloads/AirPassengers.csv"
    df = pd.read_csv(link)
    plot_df(df,x=df.index,y=df["#Passengers"],title="air passengers",xlabel="Month",ylabel="#Passengers")
    plt.show()

    # Month is string by default - convert into timestamp
    df['Month'] = pd.to_datetime(df['Month'])
    # year column
    df['year'] = [d.year for d in df.Month]

    # month column
    df['month'] = [d.strftime("%b") for d in df.Month]
    years = df['year'].unique()
    print(df)

    # lets convert the Month into date format
    df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)
    # add year column
    df['Year'] = [dt.year for dt in df.Month]

    print(df.head())

    # set Month as Index of this dataframe
    data_df = df.set_index('Month', inplace=False)
    print(data_df.head())

    # performing moving average
    ts=data_df["#Passengers"]
    ma=ts.rolling(12).mean()
    plt.plot(ts)
    plt.plot(ma)
    plt.show()


    # decopomse the time series model into different components

    from statsmodels.tsa.seasonal import seasonal_decompose
    decom=seasonal_decompose(ts)
    trend=decom.trend
    seasonality=decom.seasonal
    residual=decom.resid
    plt.subplot(411)
    plt.plot(ts,label="orginal data")
    plt.subplot(412)
    plt.plot(trend,label="trend data")
    plt.subplot(413)
    plt.plot(seasonality,label="seasonality")
    plt.subplot(414)
    plt.plot(residual,label="residual")
    plt.show()


    #testing if the data is stationary or not
    #AD Fuller test
    from statsmodels.tsa.stattools import adfuller,kpss
    ad_ful=adfuller(ts)
    stats_val = ad_ful[0]
    p_val=ad_ful[1]
    print("stats_val , p_val",stats_val,p_val)
    if p_val > 0.05:
        print(" time series is stationary")
    else:
        print("time series is not stationary")

    '''
    Time to make prediction using Time Series model. We are going to use:
    AR - I - MA
    Auto Regressive: relationship between observation and 
    specfied number of lagged observations (how many periods in the back to look for?)
    p - number of lag observations

    Integrated - the differencing factor to make the data stationary.
    d - number of times you differentiate to make the data look stationary

    Moving Average - Actual prediction of the dataset once we remove all the other components
    of the data. 
    q - how many periods I can use for find moving average

    '''
from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(ts,order=(2,1,2)) # p , d, q values
output=model.fit()
plt.plot(ts,color="black")
plt.plot(output.fittedvalues,color="red")
plt.show()

## Make predictions

from statsmodels.graphics.tsaplots import plot_predict
plot_predict(start=1,end=160,result=output)
plt.show()


## calculating the errors
pred=output.fittedvalues
mse=sum((pred-ts)**2)/len(ts)
rmse=mse**0.5
print("rmse for model 1",rmse)

## second model
model1=ARIMA(ts,order=(1,1,0)) # p,d,q values
output=model1.fit()
plt.plot(ts,color="orange")
plt.plot(output.fittedvalues,color="pink")
plt.show()

#calculating the errors
pred1=output.fittedvalues
mse=sum((pred1-ts)**2)/len(ts)
rmse=mse**0.5
print("rmse of second model",rmse)









