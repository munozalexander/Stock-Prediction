import sys, os
sys.path.append('..')
from lstm import StockModel
from helpers import *
os.chdir('..')

aapl_model = StockModel('AAPL')

# optimize lookback ==> 7
print "...tuning lookback"
lookback_range = [7, 14, 21, 28, 35]
lookback_rmse = []
for e,lookback in enumerate(lookback_range):
    aapl_model.loadStock(lookback=lookback, validation_split=True)
    model, history = aapl_model.train(lstm_dim1=128, lstm_dim2=128, dropout=0.2, dense_dim1=None, epochs=50)
    rmse = aapl_model.validate(model)
    lookback_rmse.append(rmse)
    print "%i/%i lookback tunings complete." % (e+1,len(lookback_range))
plotHyperparameterTuning(lookback_range, lookback_rmse, 'lookback')

# optimize lstm_dim1 ==> 512
print "...tuning lstm_dim1"
aapl_model.loadStock(lookback=7, validation_split=True)
lstm_dim1_range = [16, 32, 64, 128, 256, 512, 1024]
lstm_dim1_rmse = []
for e,lstm_dim1 in enumerate(lstm_dim1_range):
    model, history = aapl_model.train(lstm_dim1=lstm_dim1, lstm_dim2=128, dropout=0.2, dense_dim1=None, epochs=100)
    rmse = aapl_model.validate(model)
    lstm_dim1_rmse.append(rmse)
    print "%i/%i lstm_dim1 tunings complete." % (e+1,len(lstm_dim1_range))
plotHyperparameterTuning(lstm_dim1_range, lstm_dim1_rmse, 'lstm_dim1')

# optimize lstm_dim2 ==> 256
print "...tuning lstm_dim2"
lstm_dim2_range = [16, 32, 64, 128, 256, 512, 1024]
lstm_dim2_rmse = []
for e,lstm_dim2 in enumerate(lstm_dim2_range):
    model, history = aapl_model.train(lstm_dim1=512, lstm_dim2=lstm_dim2, dropout=0.2, dense_dim1=None, epochs=100)
    rmse = aapl_model.validate(model)
    lstm_dim2_rmse.append(rmse)
    print "%i/%i lstm_dim2 tunings complete." % (e+1,len(lstm_dim2_range))
plotHyperparameterTuning(lstm_dim2_range, lstm_dim2_rmse, 'lstm_dim2')

# optimize dropout ==> 0.3
print "...tuning dropout"
dropout_range = [0, .1, .2, .3, .4]
dropout_rmse = []
for e,dropout in enumerate(dropout_range):
    model, history = aapl_model.train(lstm_dim1=512, lstm_dim2=256, dropout=dropout, dense_dim1=None, epochs=100)
    rmse = aapl_model.validate(model)
    dropout_rmse.append(rmse)
    print "%i/%i dropout tunings complete." % (e+1,len(dropout_range))
plotHyperparameterTuning(dropout_range, dropout_rmse, 'dropout')

# optimize dense_dim1 ==> 16
print "...tuning dense_dim1"
dense_dim1_range = [0, 4, 8, 16, 32]
dense_dim1_rmse = []
for e,dense_dim1 in enumerate(dense_dim1_range):
    if dense_dim1 == 0:
        dense_dim1 = None
    model, history = aapl_model.train(lstm_dim1=512, lstm_dim2=256, dropout=0.3, dense_dim1=dense_dim1, epochs=100)
    rmse = aapl_model.validate(model)
    dense_dim1_rmse.append(rmse)
    print "%i/%i dense_dim1 tunings complete." % (e+1,len(dense_dim1_range))
plotHyperparameterTuning(dense_dim1_range, dense_dim1_rmse, 'dense_dim1')
