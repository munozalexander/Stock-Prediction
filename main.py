from lstm import StockModel

if __name__ == '__main__':
    #tickers = ['AAPL', 'AMZN', 'FB', 'GOOG', 'TSLA']
    tickers = ['AMZN', 'FB', 'GOOG']
    for ticker in tickers:
        stock_model = StockModel(ticker)
        stock_model.loadStock(lookback=40, validation_split=False)
        model, history = stock_model.train()
        stock_model.plotOneDayCurve(model)
        stock_model.plotFutureCurves(model)
        stock_model.plotBuySellPoints(model)
        stock_model.plotPortfolioReturn(model)
