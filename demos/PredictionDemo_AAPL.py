import sys, os
sys.path.append('..')
from lstm import StockModel
os.chdir('..')

aapl_model = StockModel('AAPL')
aapl_model.loadStock()
model, history = aapl_model.train()
rmse = aapl_model.validate(model)
aapl_model.plotOneDayCurve(model)
aapl_model.plotFutureCurves(model)
aapl_model.plotBuySellPoints(model)
aapl_model.plotPortfolioReturn(model)
