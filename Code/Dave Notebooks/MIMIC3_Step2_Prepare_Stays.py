
# coding: utf-8

# In[ ]:

get_ipython().magic(u'pylab inline')


# In[ ]:

import glob
import numpy as np
import os
import pandas as pd
import re
import sys
import time

from pandas import DataFrame, Series
from pandas.tslib import Timestamp


# In[ ]:

# set these as appropriate

MIMIC_PATH = '/Users/davekale/data/MIMIC_PROCESSED/'
OUTPUT_PATH = '/Users/davekale/data/mimic3_12'
VAR_BOUNDS_FILE = '/Users/davekale/data/mimic3_variables.csv'

try:
    os.makedirs(OUTPUT_PATH)
except:
    pass


# In[ ]:

variables = ['diastolic blood pressure',
           'systolic blood pressure',
           'capillary refill rate',
           'fraction inspired o2',
           'glasgow coma scale total',
           'glucose',
           'heart rate',
           'ph',
           'respiratory rate',
           'oxygen saturation',
           'temperature',
           'weight',
           'height',
           'gender',
           'race' ]

channels = variables[:-3]
channels.insert(0, 'time')

var_bounds = DataFrame.from_csv(VAR_BOUNDS_FILE, index_col='variable')


# In[ ]:

# ETCO2: haven't found yet
# Urine output: ambiguous units (raw ccs, ccs/kg/hr, 24-hr, etc.)
# Tidal volume: tried to substitute for ETCO2 but units are ambiguous

# SBP: some are strings of type SBP/DBP
def transform_sbp(df):
    v = df.VALUE.astype(str)
    idx = v.apply(lambda s: '/' in s)
    v[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)

def transform_dbp(df):
    v = df.VALUE.astype(str)
    idx = v.apply(lambda s: '/' in s)
    v[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)

# CRR: strings with brisk, <3 normal, delayed, or >3 abnormal
def transform_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan
    v[(df.VALUE == 'Normal <3 secs') | (df.VALUE == 'Brisk')] = 0
    v[(df.VALUE == 'Abnormal >3 secs') | (df.VALUE == 'Delayed')] = 1
    return v

# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def transform_fio2(df):
    v = df.VALUE.astype(float)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.VALUE>1.0)
    v[idx] = v[idx] / 100.
    return v

# GLUCOSE, PH: sometimes have ERROR as value
def transform_lab(df):
    v = df.VALUE
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v[idx] = np.nan
    return v.astype(float)

# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def transform_o2sat(df):
    v = df.VALUE.astype(float)
    idx = (v<=1)
    v[idx] = v[idx] * 100.
    return v

# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def transform_temperature(df):
    v = df.VALUE.astype(float)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'F' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'F' in s.lower()) | (v >= 79)
    v[idx] = (v[idx] - 32) * 5. / 9
    return v

# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def transform_weight(df):
    v = df.VALUE.astype(float)
    # ounces
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'oz' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'oz' in s.lower())
    v[idx] = v[idx] / 16.
    # pounds
    idx = idx | df.VALUEUOM.fillna('').apply(lambda s: 'lb' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'lb' in s.lower())
    v[idx] = v[idx] * 0.453592
    return v

# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def transform_height(df):
    v = df.VALUE.astype(float)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'in' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'in' in s.lower())
    v[idx] = np.round(v[idx] * 2.54)
    return v

var_transforms = {
    'systolic blood pressure': transform_sbp,
    'diastolic blood pressure': transform_dbp,
    'capillary refill rate': transform_crr,
    'fraction inspired o2': transform_fio2,
    'glucose': transform_lab,
    'ph': transform_lab,
    'oxygen saturation': transform_o2sat,
    'temperature': transform_temperature,
    'weight': transform_weight,
    'height': transform_height
}


# In[ ]:

S = []
T = []

patient_id = []
admit_id = []
episode_id = []
encounter_no = []
age = []
race = []
sex = []
admit_diagnosis = []
weight = []
height = []

Yd = []
ymort = []
ylos = []

dirs = glob.glob(os.path.join(MIMIC_PATH, '*'))
nb_dirs = len(dirs)
for d_no, d in enumerate(dirs):
    if not os.path.isdir(d):
        continue
    
    stays = DataFrame.from_csv(os.path.join(d, 'stays.csv'), index_col='ICUSTAY_ID', parse_dates=False).drop_duplicates()
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DEATHTIME = stays.DEATHTIME.apply(lambda s: pd.to_datetime(s) if type(s) is str or not np.isnan(s) else np.nan)
    stays.sort_values(by='INTIME', inplace=True)
    diags = DataFrame.from_csv(os.path.join(d, 'diagnoses.csv'), index_col=None, parse_dates=False).sort_values(by=['ICUSTAY_ID', 'SEQ_NUM']).set_index('ICUSTAY_ID')
    obs = DataFrame.from_csv(os.path.join(d, 'observations.csv'), index_col=None, parse_dates=False)
    obs.CHARTTIME = pd.to_datetime(obs.CHARTTIME)

    # Transform values, remove outliers
    for vname in var_transforms.keys():
        if (obs.VARIABLE==vname).any():
            obs.VALUE[obs.VARIABLE==vname] = var_transforms[vname](obs.ix[obs.VARIABLE==vname])
            v = obs.VALUE[obs.VARIABLE==vname]
            v[v < var_bounds.dropBelow[vname]] = np.nan
            v[v < var_bounds.minValue[vname]]  = var_bounds.minValue[vname]
            v[v > var_bounds.dropAbove[vname]] = np.nan
            v[v > var_bounds.maxValue[vname]]  = var_bounds.maxValue[vname]
            obs.VALUE[obs.VARIABLE==vname] = v
    obs.VALUE = obs.VALUE.astype(float)
    obs = obs.ix[obs.VALUE.notnull()]
    if obs.shape[0] == 0:
        sys.stdout.write('\rSUBJECT {0} of {1} ({2}) has no data!'.format(d_no, nb_dirs, re.search('(\d+)', d).group(1)))
        continue
    
    for stay_no, stay_id in enumerate(stays.index):
        idx = ((obs.CHARTTIME >= stays.INTIME[stay_id]) & (obs.CHARTTIME <= stays.OUTTIME[stay_id])) | (obs.ICUSTAY_ID == stay_id)
        if not idx.any():
            sys.stdout.write('\rSUBJECT {0} of {1} ({2}), STAY_ID {3}: EMPTY!'.format(d_no, nb_dirs, stays.SUBJECT_ID[stay_id], stay_id))
            continue
        
        X = obs[['CHARTTIME', 'VARIABLE', 'VALUE']].ix[idx].sort_values(by='CHARTTIME').drop_duplicates(['CHARTTIME', 'VARIABLE'])
        nb_obs = X.shape[0]
        X = X.pivot(index='CHARTTIME', columns='VARIABLE', values='VALUE').sort_index()
        for c in variables:
            if c not in X.columns:
                X[c] = np.nan

        try:
            hours = (X.index.max() - X.index.min()).total_seconds()/60./60
        except:
            continue
        else:
            if hours < 12:
                sys.stdout.write('\rSUBJECT {0} of {1} ({2}), STAY_ID {3}: <12 hours data'.format(d_no, nb_dirs, stays.SUBJECT_ID[stay_id], stay_id))
                continue
        
        patient_id.append(stays.SUBJECT_ID[stay_id])
        admit_id.append(stays.HADM_ID[stay_id])
        episode_id.append(stay_id)
        encounter_no.append(stay_no)
        age.append(stays.AGE[stay_id])
        race.append(stays.ETHNICITY[stay_id])
        sex.append(stays.GENDER[stay_id])
        admit_diagnosis.append(stays.DIAGNOSIS[stay_id])
        weight.append(X.weight[X.weight.notnull()].iloc[0] if X.weight.notnull().any() else np.nan)
        height.append(X.height[X.height.notnull()].iloc[0] if X.height.notnull().any() else np.nan)

        yd = diags.ICD9_CODE[stay_id].tolist()
        if type(yd) is not list:
            yd = [ yd ]
        Yd.append(yd)
        try:
            ym = (stays.DEATHTIME[stay_id] < stays.OUTTIME[stay_id])
        except:
            ym = False
        ymort.append(ym)
        ylos.append(stays.LOS[stay_id])

        X = X.reset_index().rename(columns={'CHARTTIME': 'time'})
        X.time = (X.time - X.time.min()) / np.timedelta64(1, 'm')
        S.append(X[channels].values)
        sys.stdout.write('\rSUBJECT {0} of {1} ({2}), STAY_ID {3}: {4} observations'.format(d_no, nb_dirs, stays.SUBJECT_ID[stay_id], stay_id, nb_obs))
print ''


# In[ ]:

admit_diagnoses = np.unique(admit_diagnosis)
sex_categories = np.array(['M', 'F'])
race_categories = np.unique(race)
diag_codes = DataFrame.from_csv(os.path.join(MIMIC_PATH, 'diagnostic_code_stats.csv'), index_col=None)
Yd_codes = diag_codes.ICD9_CODE.values.astype(str)

patient_id = np.array(patient_id).astype(int)
admit_id = np.array(admit_id).astype(int)
episode_id = np.array(episode_id).astype(int)
encounter_no = np.array(encounter_no).astype(int)
age = np.array(age)
race = (np.array(race)[:,None] == race_categories[None,:]).astype('int8')
sex = (np.array(sex)[:,None] == sex_categories[None,:]).astype('int8')
admit_diagnosis = (np.array(admit_diagnosis)[:,None] == admit_diagnoses[None,:]).astype('int8')
weight = np.array(weight)
height = np.array(height)

Yd = np.vstack([ (np.array(yd).astype(str)[:,None] == Yd_codes).any(axis=0) for yd in Yd ]).astype('int8')
ymort = np.array(ymort).astype('int8')
ylos = np.array(ylos)


# In[ ]:

fns = [ os.path.join(OUTPUT_PATH, 'mimic3_12-NO_TIME_SERIES.npz'),
        os.path.join(OUTPUT_PATH, 'mimic3_12.npz') ]
sequences = [ [], S ]
for fn, seq in zip(fns, sequences):
    st = time.time()
    np.savez(fn,X=seq,
                episode_id=episode_id,
                admit_id=admit_id,
                patient_id=patient_id,
                encounter_no=encounter_no,
                age=age,
                race=race,
                sex=sex,
                admit_diagnosis=admit_diagnosis,
                Yd=Yd,
                ymort=ymort,
                ylos=ylos,
                race_categories=race_categories,
                sex_categories=sex_categories,
                admit_diagnoses=admit_diagnoses,
                Yd_codes=Yd_codes,
                X_names=channels)
    print 'Took', time.time()-st

