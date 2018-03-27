
# coding: utf-8

# # Introduction

# Notes that were evaluated were pulled from both MIMIC 2 and MIMIC 3 due to availablity of data at the time of extraction.  This causes difficulty when trying to pair the notes with structured data for the purposes of relating the notes to the overall patient context.  
# 
# This notebook takes annotated note files as input, collects the necessary information from MIMIC 2 and 3 to find the note in MIMIC 3, and creates a new annotation file with the MIMIC 3 data.

# **Authors**
# - Eric Carlson

# In[1]:

from datetime import datetime
import configparser
import hashlib
from importlib import reload
import logging
import numpy as np
import os
import pandas as pd
import pathlib as pl
import sys
import yaml

from IPython import display

import etc_utils as eu
import mimic_extraction_utils as meu


# In[2]:

import sqlalchemy as sa
from sqlalchemy.engine import reflection
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base


# In[3]:

reload(eu)
#%cat etc_utils.py


# In[4]:

reload(meu)


# ## Configure pandas and matplot lib for nice web printing

# In[5]:

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100


# In[6]:

get_ipython().magic('matplotlib inline')


# ## Load config files, configure logging

# In[7]:

work_desc = "mimic_iii_note_conversion"


# In[24]:

time_str, path_config, creds = eu.load_config_v2(creds_file='../../private/mit_freq_fliers/credentials.yaml')
print('Time string: {}'.format(time_str))

print('Paths:')
for k, item in path_config.items():
    print('{}: {}'.format(k, item))


# In[9]:

logger = logging.getLogger()

eu.configure_logging(logger, work_desc=work_desc, log_directory=path_config['log_dir'], time_str=time_str)


# In[10]:

[k for k in creds.keys()]


# In[11]:

[k for k in creds['mimic3_v1_4'].keys()]


# # Connect to databases

# In[12]:

db3_path = '{sa_driver}://{username}:{password}@{hostname}/{dbname}'.format(**creds['mimic3_v1_4'])
engine3 = create_engine(db3_path)
sm3 = sessionmaker(bind=engine3)
s3 = sm3()
conn3 = s3.connection()

meta3 = MetaData(schema="mimiciii", bind=engine3)
meta3.reflect(bind=engine3)


# In[13]:

base3 = automap_base(metadata=meta3)
base3.prepare(engine3, reflect=True)


# In[14]:

for cls_name in sorted(base3.classes.keys()):
    print(cls_name)


# In[15]:

note_tb = base3.classes['noteevents']


# In[16]:

s3.query(note_tb.category).count()


# In[17]:

db2_path = '{sa_driver}://{username}:{password}@{hostname}/{dbname}'.format(**creds['mimic2_v2_6'])
engine2 = create_engine(db2_path)
sm2 = sessionmaker(bind=engine2)
s2 = sm2()
conn2 = s2.connection()

meta2 = MetaData(schema="mimic2v26", bind=engine2)
meta2.reflect(bind=engine2)


# In[18]:

base2 = automap_base(metadata=meta2)
base2.prepare(engine2, reflect=True)


# In[19]:

for cls_name in sorted(base2.classes.keys()):
    print(cls_name)


# In[20]:

conn2.execute('select count(*) from mimic2v26.noteevents').fetchall()


# # Load labeled notes

# In[21]:

categories = ['Advanced.Cancer', 'Advanced.Heart.Disease', 'Advanced.Lung.Disease',
       'Alcohol.Abuse',
       'Chronic.Neurological.Dystrophies', 'Chronic.Pain.Fibromyalgia',
       'Dementia', 'Depression', 'Developmental.Delay.Retardation',
       'Non.Adherence', 'None',
       'Obesity', 'Other.Substance.Abuse', 
       'Schizophrenia.and.other.Psychiatric.Disorders', 'Unsure',]


# In[25]:

data_path = pl.Path(path_config['repo_data_dir'])


# In[26]:

[p for p in data_path.glob('*csv')]


# In[27]:

nursing_notes_path = data_path.joinpath('nursingNotesClean.csv')
discharge_notes_path = data_path.joinpath('dischargeSummariesClean.csv')


# In[28]:

nursing_notes = pd.read_csv(nursing_notes_path.as_posix())
disch_notes = pd.read_csv(discharge_notes_path.as_posix()).rename(columns={'subject.id':'subject_id'})


# In[29]:

display.display(nursing_notes.head(5))
print(nursing_notes.loc[0,'text'])


# In[30]:

display.display(disch_notes.head(5))
print(disch_notes.loc[0,'text'][:500])


# # Gather those patient's data from the database and export

# From extracts above, see that nursing notes are from MIMIC II (indicated by dates, also from Slack discussion).  Discharge notes seem to be a combination, with chartdate in MIMIC III format, but MIMIC II dates in the notes themselves.
# 
# Joy (via Slack):
# 
#     When we first pulled the notes,  only MIMIC II was available. However, MIMIC II did not have very good notes pulled from the raw clinical data. In particular, lots of discharge notes were missing. Nursing notes were more decent so we started annotating the nursing notes first. Then we got Leo's people to pull discharge notes from MIMIC III for us when it became ready
# 
# Approach:
# 
# 1. Extract list of all patients (`subject_id`) from notes files
# 1. Extract those patients' note metadata: note id, text md5sum, dates, type, icustayid, hadm_id
# 1. Extract those patients' icustayid info, including dates
# 1. For each note in the notes file, try to match against a note in one of the databases
# 1. Find the MIMIC III id data (subject_id, hadm_id, icustay_id, note_id)
# 1. Output consistent file with annotations and MIMIC III metadata

# Create union of all subject IDs in either set:

# In[28]:

subject_ids = set(nursing_notes['subject_id']) | set(disch_notes['subject_id'])
len(subject_ids)


# ## Gather metadata from MIMIC II

# Extract existing metadata, as well as information that can be used for matching:
# 
# - md5 hash of original text - only useful if unchanged
# - text length, very rough matching
# - beginning and end of string, stripped of whitespace, template words (e.g. Admission Date), and de-id templates (e.g. [** <date> **])

# In[29]:

query = """
select subject_id, hadm_id, icustay_id, realtime, charttime, category, 
    md5(text) as "md5", length(text) as "length", 
    left(strip_text, 50) as "str_start", 
    right(strip_text, 50) as "str_end"
from (
    select *, regexp_replace(text, '\s|\[\*\*[^\*]+\*\*\]|Admission Date|Discharge Date|Date of Birth', '', 'g') as strip_text
    from mimic2v26.noteevents
    where category in ('Nursing/Other', 'DISCHARGE_SUMMARY')
    and subject_id in ({})
    ) as a
"""


# In[30]:

m2_notes_meta = pd.read_sql(query.format(','.join([str(sid) for sid in subject_ids])), engine2)


# In[31]:

m2_notes_meta.loc[m2_notes_meta['category'] == 'Nursing/Other','category'] =   'Nursing/other'
m2_notes_meta.loc[m2_notes_meta['category'] == 'DISCHARGE_SUMMARY','category'] =   'Discharge summary'    


# In[32]:

m2_notes_meta.head(5)


# ## Gather metadata from MIMIC III

# In[33]:

query = """
select row_id, subject_id, hadm_id, chartdate, charttime, storetime, category,
    md5(text) as "md5", length(text) as "length", 
    left(strip_text, 50) as "str_start", 
    right(strip_text, 50) as "str_end"
from (
    select *, regexp_replace(text, '\s|\[\*\*[^\*]+\*\*\]|Admission Date|Discharge Date|Date of Birth', '', 'g') as strip_text
    from mimiciii.noteevents
    where category in ('Nursing/other', 'Discharge summary')
    and subject_id in ({})
    ) as a
"""


# In[34]:

m3_notes_meta = pd.read_sql(query.format(','.join([str(sid) for sid in subject_ids])), engine3)


# In[35]:

m3_notes_meta.head(5)


# ## Try to match to notes files

# Add hash column

# In[36]:

disch_notes['md5'] = disch_notes['text'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
disch_notes['length'] = disch_notes['text'].apply(len)
disch_notes['str_start'] = disch_notes['text'].apply(lambda x: meu.clean_text(x)[:50])
disch_notes['str_end'] = disch_notes['text'].apply(lambda x: meu.clean_text(x)[-50:])
nursing_notes['md5'] = nursing_notes['text'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
nursing_notes['length'] = nursing_notes['text'].apply(len)
nursing_notes['str_start'] = nursing_notes['text'].apply(lambda x: meu.clean_text(x)[:50])
nursing_notes['str_end'] = nursing_notes['text'].apply(lambda x: meu.clean_text(x)[-50:])


# Verify unique notes present in annotation output

# In[37]:

len(set(disch_notes['md5'])) == len(disch_notes['md5'])


# In[38]:

len(nursing_notes['md5']) == len(set(nursing_notes['md5']))


# Notes files Wouldn't be unique because multiple annotators, that's ok

# Verify database also had unique notes

# In[39]:

len(m2_notes_meta['md5']) == len(set(m2_notes_meta['md5']))


# In[40]:

len(m3_notes_meta['md5']) == len(set(m3_notes_meta['md5']))


# In[41]:

len(m2_notes_meta['md5'])


# In[42]:

len(set(m2_notes_meta['md5']))


# We see that there are duplicate entries, inspect...

# In[43]:

g2 = m2_notes_meta.groupby('md5')
res = g2['length'].agg(['count', 'max', 'min']).sort_values('count', ascending=False).head(10)


# In[44]:

res.head()


# The most common repeat is only length 2, what is it?

# In[45]:

repeat_rows = m2_notes_meta.loc[m2_notes_meta['md5'] == 'e1c06d85ae7b8b032bef47e42e4c08f9']
repeat_rows.head(2)


# In[46]:

conn2.execute("select text from mimic2v26.noteevents where subject_id=154 and category='DISCHARGE_SUMMARY'").fetchall()


# Just empty (2 new lines).  What about the next most common?

# In[47]:

repeat_rows = m2_notes_meta.loc[m2_notes_meta['md5'] == 'e7ffa42fc2f47fd0e3eb1bc54283375e']
repeat_rows.head(2)


# In[48]:

conn2.execute("select text from mimic2v26.noteevents where subject_id=2905 and md5(text)='e7ffa42fc2f47fd0e3eb1bc54283375e'").fetchall()


# Un-informative repeated data.

# In[49]:

dat = [(r['subject_id'], r['md5']) for (ind, r) in m2_notes_meta.loc[:, ['subject_id', 'md5']].iterrows()]


# In[50]:

len(dat)


# In[51]:

len(set(dat))


# Even including subject_id has repeats.  Based on these, we should be able to left join the database data onto the file data, but there could be repeated rows which may confuse analysis.  To guarantee no duplicated rows we'll join including subject id, and also drop duplicates in the database dataframes.

# In[52]:

print(m2_notes_meta.shape)
m2_notes_meta.drop_duplicates(['subject_id', 'md5'], inplace=True)
print(m2_notes_meta.shape)


# In[53]:

print(m3_notes_meta.shape)
m3_notes_meta.drop_duplicates(['subject_id', 'md5'], inplace=True)
print(m3_notes_meta.shape)


# In[54]:

nursing_notes[['subject_id', 'icustay_id', 'charttime', 'md5']].head(5)


# Verify that md5 from nursing notes from python will match a md5 from the database from postgres...

# In[55]:

(m2_notes_meta['md5'] == 'dd77b97f8b5da793773ceb4c56f32753').sum()


# Extract subset of relevant columns, no duplicates

# In[56]:

nurs_notes_meta = nursing_notes.loc[:, ['subject_id', 'Hospital.Admission.ID', 'icustay_id', 
                                        'realtime', 'charttime', 
                                        'md5', 'length', 'str_start', 'str_end']]
nurs_notes_meta.drop_duplicates(inplace=True)
nurs_notes_meta.rename(columns={'Hospital.Admission.ID': 'hadm_id'}, inplace=True)
nurs_notes_meta['category'] = 'Nursing/other'
nurs_notes_meta.reset_index(inplace=True)
nurs_notes_meta.head(5)


# In[57]:

disch_notes_meta = disch_notes.loc[:, ['subject_id', 'Hospital.Admission.ID', 'icustay_id', 'Real.time', 'chartdate', 
                                       'md5', 'length', 'str_start', 'str_end']]
disch_notes_meta.drop_duplicates(inplace=True)
disch_notes_meta.rename(columns={'Hospital.Admission.ID': 'hadm_id'}, inplace=True)
disch_notes_meta['realtime'] = disch_notes['Real.time'].apply(meu.fix_date)
disch_notes_meta.loc[:, 'chartdate'] = disch_notes['chartdate'].apply(meu.fix_date)
disch_notes_meta.drop(['Real.time',], axis=1, inplace=True)
disch_notes_meta['category'] = 'Discharge summary'
disch_notes_meta.reset_index(inplace=True)
disch_notes_meta.head(5)


# In[58]:

ann_notes_meta = pd.concat([nurs_notes_meta, disch_notes_meta], axis=0)
print(ann_notes_meta.shape)
ann_notes_meta.head(5)


# In[59]:

m2_copy = m2_notes_meta.copy()
m2_copy.columns = [c if c in ['subject_id', 'md5'] else c+'_m2' for c in m2_copy.columns]
meta_with_m2 = pd.merge(ann_notes_meta, m2_copy, how='left', on=['subject_id', 'md5'])


# In[60]:

meta_with_m2.head(5)


# In[61]:

m3_copy = m3_notes_meta.copy()
m3_copy.columns = [c if c in ['subject_id', 'md5'] else c+'_m3' for c in m3_copy.columns]
meta_m2_m3 = pd.merge(meta_with_m2, m3_copy, how='left', on=['subject_id', 'md5'], suffixes=['', '_m3'])
meta_m2_m3.head(5)


# In[62]:

meta_m2_m3.shape


# In[63]:

(meta_m2_m3['category'] == 'Nursing/other').sum()


# In[64]:

(meta_m2_m3['category'] == 'Discharge summary').sum()


# In[65]:

meta_m2_m3['length_m3'].notnull().sum()


# In[66]:

meta_m2_m3['length_m2'].notnull().sum()


# In[67]:

((meta_m2_m3['category'] == 'Nursing/other') & meta_m2_m3['length_m2'].notnull()).sum()


# From this, almost all nursing notes were able to match to MIMIC 2, but nothing was able to match to MIMIC 3, and discharge summaries couldn't be matched at all.  Look into why discharge notes aren't matching to MIMIC 3...

# In[68]:

nurs_notes_meta.loc[0, 'md5'] in list(m2_notes_meta['md5'])


# In[69]:

nurs_notes_meta.loc[0, 'md5'] in list(m3_notes_meta['md5'])


# In[70]:

disch_notes_meta.loc[0, 'md5'] in list(m2_notes_meta['md5'])


# In[71]:

disch_notes_meta.loc[0, 'md5'] in list(m3_notes_meta['md5'])


# In[72]:

disch_notes_meta.loc[0, 'subject_id'] in list(m3_notes_meta['subject_id'])


# In[73]:

subject_id = disch_notes_meta.loc[0, 'subject_id']
note_length = disch_notes_meta.loc[0, 'length']
display.display(disch_notes_meta.loc[0,:])
m3_notes_meta.loc[(m3_notes_meta['subject_id'] == subject_id) &
                 (np.abs(m3_notes_meta['length']-note_length) < 200)]


# In[74]:

db_note = conn3.execute("""
select text 
from mimiciii.noteevents 
where subject_id=9973 
and category='Discharge summary'
and md5(text)='10577fde1d173468a939ce3cf19f0926'
""").fetchone()[0]
print(db_note[:500])


# In[75]:

print(disch_notes.loc[0, 'text'][:500])


# From this we see that the notes in our annotation files nearly match the notes in MIMIC III, but de-identification processes have changed.

# In[76]:

note_ind = 0
subject_id = disch_notes_meta.loc[note_ind, 'subject_id']
hadm_id = disch_notes_meta.loc[note_ind, 'hadm_id']
note_length = disch_notes_meta.loc[note_ind, 'length']
chartdate = disch_notes_meta.loc[note_ind, 'chartdate']
display.display(disch_notes_meta.loc[note_ind,:])
m3_notes_meta.loc[(m3_notes_meta['subject_id'] == subject_id)]
m3_notes_meta.loc[(m3_notes_meta['subject_id'] == subject_id) &
                 (m3_notes_meta['hadm_id'] == hadm_id)]

# m3_notes_meta.loc[(m3_notes_meta['subject_id'] == subject_id) &
#                  (m3_notes_meta['chartdate'] == chartdate)]


# In[77]:

type('test')


# In[78]:

m3_copy = m3_notes_meta.copy()
join_cols = ['subject_id', 'category']
m3_copy.columns = [c if c in join_cols else c+'_m3' for c in m3_copy.columns]
meta_m2_m3 = pd.merge(meta_with_m2, m3_copy, how='left', on=join_cols, suffixes=['', '_m3'])
meta_m2_m3['len_diff_pct'] = np.abs(meta_m2_m3['length'] - meta_m2_m3['length_m3'])/meta_m2_m3['length']
meta_m2_m3['str_start_diff'] = meta_m2_m3.apply(lambda r: meu.similar(r['str_start'], r['str_start_m3']), axis=1)
meta_m2_m3['str_end_diff'] = meta_m2_m3.apply(lambda r: meu.similar(r['str_end'], r['str_end_m3']), axis=1)
meta_m2_m3 = meta_m2_m3.loc[meta_m2_m3['len_diff_pct'].isnull() |
                           ((meta_m2_m3['len_diff_pct'] < .1) &
                           (meta_m2_m3['str_start_diff'] < .1) & (meta_m2_m3['str_end_diff'] < .2))]
#meta_m2_m3 = meta_m2_m3.loc[(meta_m2_m3['len_diff_pct'] < .05)].sort_values('len_diff_pct', ascending=False)
meta_m2_m3.head(5)


# In[79]:

meta_m2_m3.shape


# In[80]:

ann_notes_meta.shape


# In[81]:

(meta_m2_m3['category'] == 'Nursing/other').sum()


# In[82]:

(meta_m2_m3['category'] == 'Discharge summary').sum()


# In[83]:

meta_m2_m3['length_m3'].notnull().sum()


# In[84]:

meta_m2_m3['length_m2'].notnull().sum()


# In[85]:

((meta_m2_m3['category'] == 'Nursing/other') & meta_m2_m3['length_m2'].notnull()).sum()


# In[86]:

ann_notes_meta.loc[ann_notes_meta['category'] == 'Nursing/other', 'md5'].unique().shape


# In[87]:

ann_notes_meta.loc[ann_notes_meta['category'] == 'Discharge summary', 'md5'].unique().shape


# In[88]:

ann_notes_meta.loc[ann_notes_meta['category'] == 'Nursing/other', 'md5'].unique().shape


# In[89]:

meta_m2_m3.loc[meta_m2_m3['category'] == 'Discharge summary', 'md5'].unique().shape


# In[90]:

meta_m2_m3.loc[meta_m2_m3['category'] == 'Nursing/other', 'md5'].unique().shape


# Now almost all notes are matched to MIMIC 3, but have multiple potential matches.  We need to choose the best match in each group...

# In[91]:

meta_m2_m3.groupby(['subject_id', 'category', 'md5'])['length'].count().sort_values(ascending=False).head(10)


# In[92]:

meta_m2_m3.query('subject_id==808 & md5 == "74daffdc6966b4cfefd5715014a80bdc"')


# In[93]:

meta_m2_m3['total_m3_distance'] = meta_m2_m3['len_diff_pct'] + meta_m2_m3['str_start_diff'] + meta_m2_m3['str_end_diff']


# In[94]:

indexes = meta_m2_m3.groupby(['subject_id', 'category', 'md5']).apply(lambda r: r['total_m3_distance'].idxmin())


# In[95]:

best_matches = meta_m2_m3.loc[indexes]


# In[96]:

best_matches.shape


# In[97]:

ann_notes_meta.shape


# In[98]:

(best_matches['category'] == 'Nursing/other').sum()


# In[99]:

(best_matches['category'] == 'Discharge summary').sum()


# In[100]:

best_matches['length_m3'].notnull().sum()


# In[101]:

best_matches['length_m2'].notnull().sum()


# In[102]:

((best_matches['category'] == 'Nursing/other') & best_matches['length_m2'].notnull()).sum()


# In[103]:

ann_notes_meta.loc[ann_notes_meta['category'] == 'Nursing/other', 'md5'].unique().shape


# In[104]:

ann_notes_meta.loc[ann_notes_meta['category'] == 'Discharge summary', 'md5'].unique().shape


# In[105]:

best_matches.loc[best_matches['category'] == 'Discharge summary', 'md5'].unique().shape


# In[106]:

best_matches.loc[best_matches['category'] == 'Nursing/other', 'md5'].unique().shape


# In[107]:

best_matches.head()


# ## Save output

# In[108]:

cols_to_keep = [
    'subject_id', 'hadm_id', 'icustay_id',
    'category', 'chartdate', 'charttime', 'realtime',
    'length', 'md5', 'str_start', 'str_end', 
    'row_id_m3', 'chartdate_m3', 'charttime_m3', 'storetime_m3',
    'md5_m3', 'str_start_m3', 'str_end_m3',
    'len_diff_pct', 'str_start_diff', 'str_end_diff', 'total_m3_distance'
]
output_df = best_matches[cols_to_keep]
output_df.reset_index(inplace=True, drop=True)


# In[109]:

output_df.head()


# In[110]:

output_path = pl.Path(path_config['repo_data_dir']).joinpath('mimic3_note_equivs_{}.csv'.format(time_str))
output_path


# In[111]:

output_df.to_csv(output_path.as_posix(), index=False)


# ## Inspect questionable matches

# In[112]:

questionable_matches = output_df.sort_values('total_m3_distance', ascending=False).reset_index(drop=True)
questionable_matches.head(10)


# In[113]:

def compare_texts(out_row):
    query = """
    select text 
    from mimiciii.noteevents 
    where subject_id={subject_id} and row_id={row_id_m3}
    """.format(**out_row)
    mimic3_txt = conn3.execute(query).fetchone()[0]

    if out_row['category'] == 'Nursing/other':
        ann_txt = nursing_notes.loc[nursing_notes['md5'] == out_row['md5']].iloc[0]['text']
    else:
        ann_txt = disch_notes.loc[nursing_notes['md5'] == out_row['md5']].iloc[0]['text']
    
    print('MIMIC 3 text:\n{}'.format(mimic3_txt))
    print('Text from annotations:\n{}'.format(ann_txt))


# In[114]:

compare_texts(questionable_matches.iloc[0])


# In[115]:

compare_texts(questionable_matches.iloc[1])

