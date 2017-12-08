import matplotlib.pyplot as plt
import numpy as np

def prettify_ax(ax):
    ''' make an axis pretty '''
    for spine in ax.spines.itervalues():
        spine.set_visible(False)
    ax.set_frameon=True
    ax.patch.set_facecolor('#eeeeef')
    ax.grid('on', color='w', linestyle='-', linewidth=1)
    ax.tick_params(direction='out')
    ax.set_axisbelow(True)

def simple_ax(figsize=(6,4), **kwargs):
    ''' single prettified axis '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, **kwargs)
    prettify_ax(ax)
    return fig, ax

def earliest_date_after(query_date, date_list):
    ''' find the earliest date after a query date from ordered list of dates '''
    for i in range(len(date_list)):
        if query_date < date_list[i].date():
            return date_list[i].date()
    print '\nQUERY DATE ERROR WITH:', query_date, '\n'
    raise Exception('No values after query date')

def latest_date_before(query_date, date_list):
    ''' find the latest date before a query date from ordered list of dates '''
    for i in range(len(date_list)):
        if query_date < date_list[i].date():
            if i==0:
                print '\nQUERY DATE ERROR WITH:', query_date, '\n'
                raise Exception('No values before query date in list')
            return date_list[i-1].date()
    print '\nQUERY DATE ERROR WITH:', query_date, '\n'
    raise Exception('No values after query date in list; this could densensitize model')

def inv_price_transform(normalized_data, scaler):
    ''' inverse from normalized price to raw price '''
    m = scaler.mean_[0]
    s = scaler.scale_[0]
    return s*np.array(normalized_data)+m

def plotHyperparameterTuning(x, y, param_name):
    ''' plot a hyperparameter tuning curve '''
    f,a = simple_ax(figsize=(10,6))
    a.plot(x,y)
    a.set_xlabel(param_name)
    a.set_ylabel('RMSE')
    a.set_title('RMSE vs %s on Validation Set' % param_name)
    plt.savefig('hyperparameters/curves/%s.png' % param_name)
    print "%s tuning complete and curve saved to /hyperparameters/curves/\n\n" % param_name
