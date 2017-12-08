import sys, os
sys.path.append('..')
from lstm import StockModel
from helpers import *
os.chdir('..')

class MomentumAgent(StockModel):
    def _decideBuySell(self, startpoint, days_topredict, model, return_threshold):
        '''
        predict future prices and return a market decision
        - returns True: "buy long"
        - returns False: "sell short"
        - returns None: "do nothing"
        '''
        if self.y_test[startpoint-1] > self.y_test[startpoint-2]:
            return True
        elif self.y_test[startpoint-1] < self.y_test[startpoint-2]:
            return False
        else:
            return None

if __name__ == '__main__':
    tickers = ['AAPL', 'AMZN', 'FB', 'GOOG', 'TSLA']
    for ticker in tickers:
        aapl_momentum = MomentumAgent(ticker)
        aapl_momentum.loadStock()
        aapl_momentum.plotBuySellPoints(model=None, \
                                        return_threshold=None, \
                                        days_topredict=None, \
                                        filename='momentum_buysell.png')
        aapl_momentum.plotPortfolioReturn(model=None, \
                                          return_threshold=None, \
                                          days_topredict=None, \
                                          filename='momentum_return.png')
