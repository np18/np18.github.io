
# coding: utf-8

# # Introduction

# Collect relevant structured data for all labeled patients, for the purpose of creating classifiers separate from the NLP project, to identify patient cohorts to preferrentially sample for manual concept analysis.

# ## Code setup

# Import all of the standard library tools

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


# Imports for sqlalchemy

# In[2]:

import sqlalchemy as sa
from sqlalchemy.engine import reflection
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base, name_for_scalar_relationship, generate_relationship


# Import the ICD9 library

# In[3]:

sys.path.append('icd9')
from icd9 import ICD9

tree = ICD9('icd9/codes.json')


# Import our utility code

# In[4]:

import etc_utils as eu
import mimic_extraction_utils as meu
import structured_data_utils as sdu


# Reload block that can be run as utility code is modified

# In[5]:

reload(eu)
reload(meu)
reload(sdu)


# ## Configure pandas and matplot lib for nice web printing

# In[6]:

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100


# In[7]:

get_ipython().magic('matplotlib inline')


# ## Load config files, configure logging

# In[8]:

work_desc = "gather_structured_data"


# In[9]:

time_str, path_config, creds = eu.load_config_v2(creds_file='../../private/mit_freq_fliers/credentials.yaml')
print('Time string: {}'.format(time_str))

print('Paths:')
for k, item in path_config.items():
    print('{}: {}'.format(k, item))


# In[10]:

logger = logging.getLogger()

eu.configure_logging(logger, work_desc=work_desc, log_directory=path_config['log_dir'], time_str=time_str)


# In[11]:

[k for k in creds.keys()]


# # Connect to database

# Here we'll use the credentials loaded from our config file to connect to the MIMIC postgres database

# In[12]:

db_path = '{sa_driver}://{username}:{password}@{hostname}/{dbname}'.format(**creds['mimic3_v1_4'])
engine = create_engine(db_path)
sm = sessionmaker(bind=engine)
s = sm()
conn = s.connection()

meta = MetaData(schema="mimiciii", bind=engine)
meta.reflect(bind=engine)


# SQLAlchemy has a handy database reflection capability we can use to create access points to all of the MIMIC tables

# In[12]:

base = automap_base(metadata=meta)
base.prepare(engine, reflect=True)


# Now we can quickly check out the schema it discovered and test it

# In[13]:

for cls_name in sorted(base.classes.keys()):
    print(cls_name)


# In[14]:

note_tb = base.classes['noteevents']


# In[15]:

s.query(note_tb.category).count()


# # Load labeled notes

# Load the annotation files to get the patient IDs we need to extract ICD9 data for.  Also, use our patient matching code from previous notebook to find MIMIC III equivalent IDs for the patients.

# In[13]:

categories = ['Advanced.Cancer', 'Advanced.Heart.Disease', 'Advanced.Lung.Disease',
       'Alcohol.Abuse',
       'Chronic.Neurological.Dystrophies', 'Chronic.Pain.Fibromyalgia',
       'Dementia', 'Depression', 'Developmental.Delay.Retardation',
       'Non.Adherence', 'None',
       'Obesity', 'Other.Substance.Abuse', 
       'Schizophrenia.and.other.Psychiatric.Disorders', 'Unsure',]


# In[14]:

data_path = pl.Path(path_config['repo_data_dir'])


# In[15]:

[p for p in data_path.glob('*csv')]


# In[19]:

nursing_notes_path = data_path.joinpath('nursingNotesClean.csv')
discharge_notes_path = data_path.joinpath('dischargeSummariesClean.csv')


# In[20]:

nursing_notes = pd.read_csv(nursing_notes_path.as_posix())
disch_notes = pd.read_csv(discharge_notes_path.as_posix()).rename(columns={'subject.id':'subject_id'})


# In[21]:

display.display(nursing_notes.head(1))
print(nursing_notes.loc[0,'text'])


# In[22]:

display.display(disch_notes.head(1))
#print(disch_notes.loc[0,'text'])


# In[23]:

mimic3_map = pd.read_csv(data_path.joinpath('mimic3_note_equivs_2016-10-22-03-39.csv').as_posix())


# In[24]:

mimic3_map.head(5)


# In[25]:

disch_notes['md5'] = disch_notes['text'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
nursing_notes['md5'] = nursing_notes['text'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())


# In[26]:

cols_to_keep = ['subject_id', 'category', 'md5', 'operator'] + categories
comb_dat = pd.concat([disch_notes[cols_to_keep], nursing_notes[cols_to_keep]])


# In[27]:

comb_dat.head()


# In[28]:

comb_dat.tail()


# In[29]:

comb_dat = pd.merge(comb_dat, mimic3_map[['md5', 'row_id_m3', 'total_m3_distance']], on='md5', how='left')


# In[30]:

comb_dat.head()


# In[31]:

grouped = comb_dat.groupby('md5')['total_m3_distance'].last()
pct_unmatched = grouped.isnull().sum()/grouped.count() * 100
print('{:3.2}% of notes had no exact match in MIMIC 3'.format(pct_unmatched))


# In[56]:

output_path = pl.Path(path_config['repo_data_dir']).joinpath('combined_label_data_{}.csv'.format(time_str))
logger.info(output_path)
comb_dat.to_csv(output_path.as_posix(), index=False)


# # Gather patient's data from the database and export

# ## Test functions for gathering data

# In[33]:

note_meta = sdu.get_note_metadata(conn, comb_dat.loc[2, 'row_id_m3'])


# In[34]:

for (k, v) in note_meta.items():
    print("{}:\t{}".format(k, v))


# In[35]:

note_meta.keys()


# In[36]:

diags = sdu.get_hadm_diagnoses(conn, note_meta['hadm_id'])


# In[37]:

diags[:3]


# In[38]:

[d['clean_icd9_code'] for d in diags if d['known_icd9_code']]


# In[39]:

sdu.print_icd9_tree(diags[0]['clean_icd9_code'])


# In[40]:

sdu.print_icd9_tree(diags[5]['clean_icd9_code'])


# In[41]:

sdu.print_icd9_tree(diags[7]['clean_icd9_code'])


# In[42]:

sdu.print_icd9_tree(diags[8]['clean_icd9_code'])


# In[43]:

reload(sdu)


# In[44]:

sdu.print_icd9_tree(diags[8]['clean_icd9_code'])


# In[45]:

sdu.get_icd9_levels(diags[8]['clean_icd9_code'], 5)


# ## Gather the data

# Reload any changes made to the structured data utils code

# In[46]:

reload(sdu)


# Use pandas to extract the list of unique notes and patients - the primary thing we're looking for is the MIMIC III row id, which is used to get the MIMIC encounter ID, and from there the ICD9 diagnoses.

# In[47]:

found_notes = comb_dat.loc[comb_dat['row_id_m3'].notnull()].    groupby(['subject_id', 'md5', 'row_id_m3']).count()['total_m3_distance'].index.tolist()


# Iterate through the rows, building up a dictionary of dictionaries.  `note_info` is a dictionary where the keys are the unique subject_id-md5-row_id triplet from the pandas line above.  The values are another dictionary with 2 keys: 
# 
#   * `meta` - note metadata, including the patient id (`subject_id`), encounter id (`hadm_id`), and associated timestamps
#   * `diagnoses` - a list of the diagnoses associated with this encounter, including the original poorly formated ICD9 code from MIMIC, the reformated version (`clean_icd9_code`), and the label of the code

# In[48]:

note_info = {}
for idx in found_notes:
    note_meta = sdu.get_note_metadata(conn, idx[2])
    note_diag = sdu.get_hadm_diagnoses(conn, note_meta['hadm_id'])
    dat = {'meta': note_meta, 'diagnoses': note_diag}
    note_info[idx] = dat


# Print one element out to see how it looks

# In[49]:

note_info[[k for k in note_info.keys()][0]]


# Now we can use this list and the ICD9 python library to extract all of the parents for each code.  Not all codes in MIMIC III are know to the library (likely due to slightly different ICD versions), so we need to handle that possibility by just skipping unknown codes.  If it's a know code then we look up the parents, and we'll add the code and each of its parents to the `note_codes` list.  We'll also keep a list of the metadata.

# In[ ]:

note_codes = []
note_meta = []
unknown_codes = set()
for k, note_dat in note_info.items():
    subject_id, md5, row_id = k

    meta = note_dat['meta'].copy()
    meta['subject_id'] = subject_id
    meta['md5'] = md5
    meta['note_row_id'] = row_id
    note_meta.append(meta)

    diagnoses = note_dat['diagnoses']
    if diagnoses is not None:
        for diag in diagnoses:
            new_code = {
                'subject_id': subject_id,
                'md5': md5,
                'note_row_id': row_id,
                'level': 'source',
                'code': diag['icd9_code']
            }
            note_codes.append(new_code)

            if diag['known_icd9_code']:
                levels = sdu.get_icd9_levels(diag['clean_icd9_code'])
                for ind, lev_code in enumerate(levels):
                    new_code = {
                        'subject_id': subject_id,
                        'md5': md5,
                        'note_row_id': row_id,
                        'level': ind,
                        'code': lev_code
                    }
                    note_codes.append(new_code)

            else:
                if diag['icd9_code'] not in unknown_codes:
                    unknown_codes.add(diag['icd9_code'])
                    logger.info('Unknown code ({}) for subject ({})'.format(diag['icd9_code'], subject_id))


# In[51]:

len(unknown_codes)


# Inspecting the records, we see that for a particular note (row id 1414073), the code found a known ICD9 code (39891), then found a root parent (390-459), and the path from it through children 393-398, 398, ...  We keep track of the hierarchy level from the root node - in a future post we'll use this info to select a cutoff depth for classification based on ICD9

# In[52]:

note_codes_df = pd.DataFrame.from_records(note_codes)
note_codes_df.head(5)


# In[57]:

output_path = pl.Path(path_config['repo_data_dir']).joinpath('notes_icd9_codes_{}.csv'.format(time_str))
logger.info(output_path)
note_codes_df.to_csv(output_path.as_posix(), index=False)


# In[54]:

note_meta_df = pd.DataFrame.from_records(note_meta)
note_meta_df.head(5)


# In[58]:

output_path = pl.Path(path_config['repo_data_dir']).joinpath('mimic3_note_metadata_{}.csv'.format(time_str))
logger.info(output_path)
note_meta_df.to_csv(output_path.as_posix(), index=False)

