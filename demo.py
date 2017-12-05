from lstm import Stock_Model

aapl_model = Stock_Model('AAPL')
aapl_model.loadStock()
model, history = aapl_model.train()
rmse = aapl_model.validate(model)
aapl_model.plotOneDayCurve(model)
aapl_model.plotFutureCurves(model)
aapl_model.plotBuySellPoints(model)
