from lstm import StockModel

if __name__ == '__main__':
    #tickers = ['AAPL', 'AMZN', 'FB', 'GOOG', 'TSLA']
    tickers = ['FB', 'AMZN', 'GOOG']
    for ticker in tickers:
        stock_model = StockModel(ticker)
        stock_model.loadStock(lookback=30, validation_split=False)
        model, history = stock_model.train(lstm_dim1=256,dense_dim1=16)
        stock_model.plotOneDayCurve(model)
        stock_model.plotFutureCurves(model)
        stock_model.plotBuySellPoints(model)
        stock_model.plotPortfolioReturn(model)
