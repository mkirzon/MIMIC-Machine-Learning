import numpy as np
import os
import pandas as pd
import sys
import time

from pandas import DataFrame, TimedeltaIndex, Series
from pandas.tslib import Timestamp
#from itertools import chain
#from progressbar import ProgressBar

### set paths to lookup data ###
MIMIC_PATH = '/Users/Mark/Downloads/MIMIC Data/Original'
VAR_BOUNDS_FILE = '/Users/Mark/Documents/GitHub/MIMIC-Machine-Learning/Reference Files/mimic3_variables.csv'
OUTPUT_PATH = '/Users/Mark/Downloads/MIMIC Data/Processed'

try:
    os.makedirs(OUTPUT_PATH)
except:
    pass

var_bounds = DataFrame.from_csv(VAR_BOUNDS_FILE, index_col='variable')

st = time.time()
sys.stdout.write('Loading data...')
sys.stdout.flush()
ndata = np.load(os.path.join(OUTPUT_PATH, 'mimic3_12.npz'))
X = ndata['X']
X_names = ndata['X_names']

episode_id = ndata['episode_id']
admit_id = ndata['admit_id']
patient_id = ndata['patient_id']
encounter_no = ndata['encounter_no']
age = ndata['age']
race = ndata['race']
sex = ndata['sex']
admit_diagnosis = ndata['admit_diagnosis']
race_categories = ndata['race_categories']
sex_categories = ndata['sex_categories']
admit_diagnoses = ndata['admit_diagnoses']

ylos = ndata['ylos']

print('Took {0} secs'.format(time.time()-st))
### do some additional data processing ###

X_hr = []
Ximp_hr = []

print('Processing data...')
#bar = ProgressBar()
st = time.time()
#for xid in bar(range(X.shape[0])):
for xid in range(X.shape[0]):
    # timestamps
    x = DataFrame(X[xid], columns=X_names)
    x.time = TimedeltaIndex(x.time, 'm')
    x.set_index('time', inplace=True)

    # create hourly resampled
    x_hr = x.resample('1H', how='mean', closed='left')
    assert(x_hr.index[0].total_seconds() == 0)
    ximp_hr = x_hr.combine_first(x.resample('1H', how='last', fill_method='ffill', closed='left'))
    assert(ximp_hr.index[0].total_seconds() == 0)
    for c in np.intersect1d(ximp_hr.columns, var_bounds.index):
        v = ximp_hr[c]
        v[v.isnull()] = var_bounds.imputeValue[c]
        ximp_hr[c] = v
    assert(ximp_hr.notnull().all().all())

    X_hr.append(x_hr.values)
    Ximp_hr.append(ximp_hr.values)

print('Took', time.time()-st)
