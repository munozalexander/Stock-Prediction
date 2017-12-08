from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from sklearn import preprocessing
from datetime import datetime, timedelta
import time
from helpers import *

class StockModel():
    def __init__(self, \
                 ticker, \
                 stock_file = 'data/stock/prices-split-adjusted.csv', \
                 news_directory = 'data/news/', \
                 econ_file = 'data/market/economic_indicators.csv', \
                 reddit_file = 'data/market/reddit_sentiments.csv'):
        self.ticker = ticker
        self.__stockFile = stock_file
        self.__newsDirectory = news_directory
        self.__econFile = econ_file
        self.__redditFile = reddit_file

    def __loadData(self):
        ''' merge price, company sentiment, market sentiment into one dataframe '''
        # load data
        stock_df = pd.read_csv(self.__stockFile, index_col=0)
        stock_df = stock_df[stock_df.symbol==self.ticker].close
        stock_df.index = pd.to_datetime(stock_df.index)
        news_df = pd.read_csv(self.__newsDirectory+self.ticker+'.csv', index_col=0)
        news_df.index = pd.to_datetime(news_df.index)
        econ_df = pd.read_csv(self.__econFile, index_col=0)
        econ_df.index = pd.to_datetime(econ_df.index)
        reddit_df = pd.read_csv(self.__redditFile, index_col=0)
        reddit_df.index = pd.to_datetime(reddit_df.index)
        return_df = pd.DataFrame(columns=[stock_df.name]+['stock_'+a for a in list(news_df.columns)]+\
                                 list(econ_df.columns)+['market_'+a for a in list(reddit_df.columns)])

        # clip price data that doesn't have news coverage or reddit coverage
        d0, d1 = news_df.index[0].date(), news_df.index[1].date()
        startdate = d0-(d1-d0)
        stock_df = stock_df.loc[startdate:]

        # iterate through rows, aggregating all data and appending to return_df
        for row_num in range(stock_df.shape[0]):
            new_row = []
            stock_date = stock_df.index[row_num].date()
            new_row += [stock_df.iloc[row_num]]
            new_row += list(news_df.loc[earliest_date_after(stock_date, news_df.index),:])
            new_row += list(econ_df.loc[latest_date_before(stock_date, econ_df.index),:])
            new_row += list(reddit_df.loc[earliest_date_after(stock_date, reddit_df.index),:])
            return_df.loc[stock_date] = new_row
            if row_num % 100 == 0:
                print "%i/%i rows done." % (row_num, stock_df.shape[0]),
        print "\n%s dataframe prepped. %i timepoints, each with %i features." % \
              (self.ticker, return_df.shape[0], return_df.shape[1])
        return return_df

    def loadStock(self, lookback=25, validation_split=True):
        ''' load and scale data, split into training/validation/test sets '''
        print "\n\n...loading %s stock" % self.ticker
        df = self.__loadData()
        data = df.values
        if validation_split:
            n_train = list(df.index).index(latest_date_before(df.index[-1]+timedelta(-500), pd.to_datetime(df.index)))
            n_valid = list(df.index).index(latest_date_before(df.index[-1]+timedelta(-365), pd.to_datetime(df.index)))
        else:
            n_train = list(df.index).index(latest_date_before(df.index[-1]+timedelta(-365), pd.to_datetime(df.index)))
        self.scaler = preprocessing.StandardScaler() #normalize mean-zero, unit-variance
        self.scaler.fit(data[:n_train,:])
        data = self.scaler.transform(data)
        dataX, dataY = [], []
        for timepoint in range(data.shape[0]-lookback):
            dataX.append(data[timepoint:timepoint+lookback,:])
            dataY.append(data[timepoint+lookback,0])
        if validation_split:
            self.X_train, self.X_valid, self.X_test = np.array(dataX[:n_train]), \
                                                      np.array(dataX[n_train:n_valid]), \
                                                      np.array(dataX[n_valid:])
            self.y_train, self.y_valid, self.y_test = np.array(dataY[:n_train]), \
                                                      np.array(dataY[n_train:n_valid]), \
                                                      np.array(dataY[n_valid:])
        else:
            self.X_train, self.X_test = np.array(dataX[:n_train]), \
                                        np.array(dataX[n_train:])
            self.y_train, self.y_test = np.array(dataY[:n_train]), \
                                        np.array(dataY[n_train:])
        print "Data normalized and split."

    def __buildModel(self, lstm_dim1, lstm_dim2, dropout, dense_dim1):
        ''' build keras model '''
        model = Sequential()
        model.add(LSTM(lstm_dim1, input_shape=(self.X_train.shape[1],self.X_train.shape[2]), return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(lstm_dim2, return_sequences=False))
        model.add(Dropout(dropout))
        if dense_dim1 is not None:
            model.add(Dense(dense_dim1, kernel_initializer="uniform", activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def __fitModel(self, model, epochs):
        ''' fit model to training data '''
        history = model.fit(
                    self.X_train, \
                    self.y_train, \
                    batch_size=512,
                    epochs=epochs,
                    validation_split=0,
                    verbose=0)
        return history

    def train(self, lstm_dim1=128, lstm_dim2=128, dropout=0.2, dense_dim1=None, epochs=200):
        ''' build and train model '''
        t0 = time.time()
        print "\n\n...beginning training"
        model = self.__buildModel(lstm_dim1, lstm_dim2, dropout, dense_dim1)
        history = self.__fitModel(model, epochs)
        print "TRAINING DONE. %i seconds to train.\n\n" % int(time.time()-t0)
        return model, history

    def validate(self, model):
        ''' run one-day lookup and return rmse if validate or predictions if test '''
        print "\n\n...validating"
        predictions = model.predict(self.X_valid)
        rmse = np.sqrt(np.mean((predictions-self.y_valid)**2))
        print "Validation complete with RMSE of:", rmse
        return rmse

    def __predictDays(self, startday, days_topredict, model):
        ''' starting from startday predict days_topredict stock prices '''
        curr_data = self.X_test[startday,:,:]
        predictions = []
        for day in range(days_topredict):
            prediction = model.predict(curr_data.reshape(1,curr_data.shape[0],curr_data.shape[1]))[0][0]
            predictions.append(prediction)
            new_row = curr_data[-1,:]
            new_row[0] = prediction
            curr_data = np.vstack((curr_data[1:,:], new_row))
        return predictions

    def plotOneDayCurve(self, model, filename='onedaycurve0.png'):
        ''' predict one day in future on test set and print '''
        print "\n\n...plotting one-day lookahead curve"
        predictions = model.predict(self.X_test)
        f, a = simple_ax(figsize=(10,6))
        a.plot(predictions, c='b', label='predictions')
        a.plot(self.y_test, c='r', label='actual')
        a.set_ylabel('Normalized closing price')
        a.set_xlabel('Day')
        a.set_title('%s Test Set Predictions'%self.ticker)
        plt.legend()
        plt.savefig('figures/lstm/'+self.ticker+'_'+filename)
        print "One-day lookahead curve successfully plotted and saved."

    def plotFutureCurves(self, model, days_topredict=30, filename='futurecurves0.png'):
        ''' predict future days and plot curves on test set '''
        print "\n\n...plotting future curves"
        f, a = simple_ax(figsize=(10,6))
        a.plot(inv_price_transform(self.y_test, self.scaler), c='k')
        for segment in range(int(len(self.y_test)/days_topredict)):
            predictions = self.__predictDays(segment*days_topredict, days_topredict, model)
            a.plot(range(segment*days_topredict, segment*days_topredict+days_topredict), \
                   inv_price_transform(predictions, self.scaler))
            a.axvline(segment*days_topredict, c='k', linestyle='dashed', linewidth=1)
            a.axvline(segment*days_topredict+days_topredict, c='k', linestyle='dashed', linewidth=1)
        a.set_xlabel('Day')
        a.set_ylabel('Price')
        a.set_title('%s Test Set %i Day Lookahead' % (self.ticker, days_topredict))
        plt.savefig('figures/lstm/'+self.ticker+'_'+filename)
        print "Future Curves successfully plotted and saved."

    def _decideBuySell(self, startpoint, days_topredict, model, return_threshold):
        '''
        predict future prices and return a market decision
        - returns True: "buy long"
        - returns False: "sell short"
        - returns None: "do nothing"
        '''
        predictions = self.__predictDays(startpoint, days_topredict, model)
        startprice, maxprice, minprice = predictions[0], max(predictions), min(predictions)
        buyreturn = (maxprice-startprice)/startprice
        sellreturn = (startprice-minprice)/startprice
        if buyreturn>=sellreturn and buyreturn>=return_threshold:
            return True
        elif sellreturn>buyreturn and sellreturn>=return_threshold:
            return False
        return None

    def __walkBuySell(self, days_topredict, model, return_threshold):
        ''' walk data making buy/sell decisions '''
        buy_dates, sell_dates = [], []
        for t in range(len(self.y_test)):
            decision = self._decideBuySell(t, days_topredict, model, return_threshold)
            if decision is True:
                buy_dates.append(t)
            elif decision is False:
                sell_dates.append(t)
            if t%20==0:
                print "%i/%i timepoints calculated." % (t+1,len(self.y_test)),
        print "Data walk complete."
        return buy_dates, sell_dates

    def plotBuySellPoints(self, model, return_threshold=0.5, days_topredict=30, filename='buysell0.png'):
        ''' plot points to buy or sell stock '''
        print "\n\n...plotting buy-sell point graph"
        buy_dates, sell_dates = self.__walkBuySell(days_topredict, model, return_threshold)
        f,a = simple_ax(figsize=(10,6))
        a.plot(inv_price_transform(self.y_test, self.scaler), c='k')
        a.scatter(buy_dates, inv_price_transform(self.y_test[buy_dates],self.scaler), c='g')
        a.scatter(sell_dates, inv_price_transform(self.y_test[sell_dates],self.scaler), c='r')
        a.set_xlabel('Day')
        a.set_ylabel('Price')
        a.set_title('Buy/Sell Decisions for %s Test Set' % self.ticker)
        recs = [mpatches.Rectangle((0,0),1,1,fc='g'), mpatches.Rectangle((0,0),1,1,fc='r')]
        a.legend(recs,['buy', 'sell'], loc=2, prop={'size':14})
        plt.savefig('figures/lstm/'+self.ticker+'_'+filename)
        print "Buy-sell decision points successfully plotted and saved."

    def plotPortfolioReturn(self, model, initial_cash=10000, per_trade_value=500,\
                            return_threshold=0.5, days_topredict=30, filename='portfolio0.png'):
        ''' walk the test set buying and selling, plot portfolio value over time '''
        print "\n\n...plotting portfolio return over time"
        buy_dates, sell_dates = self.__walkBuySell(days_topredict, model, return_threshold)
        cash = initial_cash
        stocks_per_trade = max([int(round(per_trade_value/self.y_test[0])), 1])
        portfolio =  0
        returns = [0]
        for date in range(len(self.y_test)):
            if date in buy_dates: #buy
                portfolio += stocks_per_trade
                cash = cash - stocks_per_trade*inv_price_transform(self.y_test[date], self.scaler)
            elif date in sell_dates: #sell
                portfolio -= stocks_per_trade
                cash = cash + stocks_per_trade*inv_price_transform(self.y_test[date], self.scaler)
            curr_value = cash + portfolio*inv_price_transform(self.y_test[date], self.scaler)
            curr_return = 100*(curr_value-initial_cash)/initial_cash
            returns.append(curr_return)
        f,a = simple_ax(figsize=(10,6))
        a.plot(returns, linewidth=2)
        a.set_xlabel('Day')
        a.set_ylabel('Portfolio Percent Return')
        a.set_title('Portfolio Value Over Time Trading %s on Test Set' % self.ticker)
        plt.savefig('figures/lstm/'+self.ticker+'_'+filename)
        print "Portfolio return graph successfully plotted and saved."
