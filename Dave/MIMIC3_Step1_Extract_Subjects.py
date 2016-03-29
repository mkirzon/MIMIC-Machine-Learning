
# coding: utf-8

# In[ ]:

get_ipython().magic(u'pylab inline')


# In[ ]:

import csv
import numpy as np
import os
import pandas as pd
import sys

from collections import defaultdict
from pandas import DataFrame


# In[ ]:

# set these as appropriate

MIMIC_PATH = '/Users/davekale/data/MIMIC3'
VARIABLE_MAP_FILE = '/Users/davekale/data/mimic3_map.csv'
OUTPUT_PATH = '/Users/davekale/data/MIMIC_PROCESSED'

try:
    os.makedirs(OUTPUT_PATH)
except:
    pass


# In[ ]:

OBS_HEADER = [ 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'VARIABLE', 'VALUE', 'VALUEUOM', 'MIMIC_LABEL' ]
NB_ROWS_CHARTEVENTS = 263201376
NB_ROWS_LABEVENTS = 27872576
NB_ROWS_OUTPUTEVENTS = 4349340


# In[ ]:

pats = DataFrame.from_csv(os.path.join(MIMIC_PATH, 'PATIENTS_DATA_TABLE.csv'))
pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
pats.DOB = pd.to_datetime(pats.DOB)
pats.DOD = pd.to_datetime(pats.DOD)

admits = DataFrame.from_csv(os.path.join(MIMIC_PATH, 'ADMISSIONS_DATA_TABLE.csv'))
admits = admits[['SUBJECT_ID', 'HADM_ID', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)

stays  = DataFrame.from_csv(os.path.join(MIMIC_PATH, 'ICUSTAYS_DATA_TABLE.csv'))
print 'START:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]

stays = stays.ix[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
stays = stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]
stays.INTIME = pd.to_datetime(stays.INTIME)
stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
print 'REMOVE ICU TRANSFERS:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]

stays = stays.merge(admits, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])
stays = stays.merge(pats, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


# In[ ]:

to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
to_keep = to_keep.ix[to_keep.ICUSTAY_ID==1][['HADM_ID']]
stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
print 'REMOVE MULTIPLE STAYS PER ADMIT:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]


# In[ ]:

stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
stays.AGE[stays.AGE<0] = 90
stays = stays.ix[stays.AGE >= 18]
print 'REMOVE PATIENTS AGE < 18:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]


# In[ ]:

diag_codes = DataFrame.from_csv(os.path.join(MIMIC_PATH, 'D_ICD_DIAGNOSES_DATA_TABLE.csv')).set_index('ICD9_CODE')
diagnoses = DataFrame.from_csv(os.path.join(MIMIC_PATH, 'DIAGNOSES_ICD_DATA_TABLE.csv'))
diagnoses = diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(),
                            left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])
diagnoses = diagnoses.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'])
diag_codes['COUNT'] = diagnoses.ICD9_CODE.value_counts()
diag_codes.sort_values('COUNT', ascending=False, inplace=True)
diag_codes = diag_codes.ix[diag_codes.COUNT.notnull()]
diag_codes.to_csv(os.path.join(OUTPUT_PATH, 'diagnostic_code_stats.csv'), index_label='ICD9_CODE')


# In[ ]:

# uncomment when testing

stay_idx = np.random.randint(0, high=stays.shape[0], size=1000)
stays = stays.iloc[stay_idx]


# In[ ]:

nb_subjects = stays.SUBJECT_ID.unique().shape[0]
for i, subject_id in enumerate(stays.SUBJECT_ID.unique()):
    sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i, nb_subjects))
    dn = os.path.join(OUTPUT_PATH, str(subject_id))
    try:
        os.makedirs(dn)
    except:
        pass
    
    stays.ix[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'), index=False)
    diagnoses.ix[diagnoses.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID','SEQ_NUM']).to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)
    f = open(os.path.join(dn, 'observations.csv'), 'w')
    f.write(','.join(OBS_HEADER) + '\n')
    f.close()
print 'DONE!'


# In[ ]:

mp = DataFrame.from_csv(VARIABLE_MAP_FILE, index_col=None).fillna('').astype(str)
mp = mp.ix[(mp.VARIABLE != '') & (mp.COUNT>0)]
mp = mp.ix[mp.VARIABLE.apply(lambda s: not s.startswith('ZZZZ'))]
mp = mp.set_index('ITEMID')


# In[ ]:

subject_ids = set([ str(s) for s in stays.SUBJECT_ID.unique() ])

tables = [ 'chartevents', 'labevents' ] #, 'outputevents' ]
nb_rows = [ NB_ROWS_CHARTEVENTS, NB_ROWS_LABEVENTS ] #, NB_ROWS_OUTPUTEVENTS ]

for table, nbr in zip(tables, nb_rows):
    r = csv.DictReader(open(os.path.join(MIMIC_PATH, table.upper() + '_DATA_TABLE.csv'), 'r'))
    curr_subject_id = ''
    last_write_no = 0
    last_write_nb_rows = 0
    last_write_subject_id = ''
    curr_obs = []
    for i,row_in in enumerate(r):
        if last_write_no != '':
            sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...last write '
                             '({3}) {4} rows for subject {5}'.format(table, i, nbr, last_write_no,
                                                                     last_write_nb_rows, last_write_subject_id))
        else:
            sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...'.format(table, i, nbr))

        subject_id = row_in['SUBJECT_ID']
        itemid = row_in['ITEMID']
        
        if subject_id not in subject_ids or itemid not in mp.index:
            continue

        row_out = { 'SUBJECT_ID': subject_id,
                    'HADM_ID': row_in['HADM_ID'],
                    'CHARTTIME': row_in['CHARTTIME'],
                    'VARIABLE': mp.VARIABLE[row_in['ITEMID']],
                    'MIMIC_LABEL': mp.LABEL[row_in['ITEMID']],
                    'VALUE': row_in['VALUE'],
                    'VALUEUOM': row_in['VALUEUOM'] }
        
        try:
            row_out['ICUSTAY_ID'] = row_in['ICUSTAY_ID']
        except:
            row_out['ICUSTAY_ID'] = ''

        if curr_subject_id != '' and curr_subject_id != subject_id:
            last_write_no += 1
            last_write_nb_rows = len(curr_obs)
            last_write_subject_id = curr_subject_id
            fn = os.path.join(OUTPUT_PATH, str(curr_subject_id), 'observations.csv')
            w = csv.DictWriter(open(fn, 'a'), fieldnames=OBS_HEADER, quoting=csv.QUOTE_MINIMAL)
            w.writerows(curr_obs)
            curr_obs = []

        curr_obs.append(row_out)
        curr_subject_id = subject_id

    if curr_subject_id != '':
        last_write_no += 1
        last_write_nb_rows = len(curr_obs)
        last_write_subject_id = curr_subject_id
        fn = os.path.join(OUTPUT_PATH, str(curr_subject_id), 'observations.csv')
        w = csv.DictWriter(open(fn, 'a'), fieldnames=OBS_HEADER, quoting=csv.QUOTE_MINIMAL)
        w.writerows(curr_obs)
        curr_obs = []
    
    del r
    sys.stdout.write('\rprocessing {0}: ROW {1} of {2}...last write '
                     '({3}) {4} rows for subject {5}'.format(table, i, nbr, last_write_no,
                                                             last_write_nb_rows, last_write_subject_id))
    print ' DONE!'

