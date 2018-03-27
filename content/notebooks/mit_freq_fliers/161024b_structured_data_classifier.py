
# coding: utf-8

# # Introduction

# Exploration of structured data and labels

# **Authors**
# - Eric Carlson

# In[1]:

structured_collection_date = '2016-10-24-16-35'


# In[2]:

from datetime import datetime
import configparser
import hashlib
from importlib import reload
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib as pl
import pickle as pkl
import sklearn as sk
from sklearn import linear_model, metrics, model_selection
import sys
import yaml

from IPython import display

import etc_utils as eu
import mimic_extraction_utils as meu
import structured_data_utils as sdu


# In[3]:

np.random.seed(12345)


# In[4]:

reload(eu)
reload(meu)
reload(sdu)


# In[6]:

sys.path.append('icd9')
from icd9 import ICD9

# feel free to replace with your path to the json file
tree = ICD9('icd9/codes.json')


# ## Configure pandas and matplot lib for nice web printing

# In[7]:

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100


# In[8]:

get_ipython().magic('matplotlib inline')


# ## Load config files, configure logging

# In[9]:

work_desc = "gather_structured_data"


# In[10]:

time_str, path_config, creds = eu.load_config_v2(creds_file='../../private/mit_freq_fliers/credentials.yaml')
print('Time string: {}'.format(time_str))

print('Paths:')
for k, item in path_config.items():
    print('{}: {}'.format(k, item))


# In[11]:

logger = logging.getLogger()

eu.configure_logging(logger, work_desc=work_desc, log_directory=path_config['log_dir'], time_str=time_str)


# In[12]:

[k for k in creds.keys()]


# # Load labeled notes

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


# In[16]:

[p for p in data_path.glob('*{}*csv'.format(structured_collection_date))]


# In[17]:

labels_path = data_path.joinpath('combined_label_data_{}.csv'.format(structured_collection_date))
note_meta_path = data_path.joinpath('mimic3_note_metadata_{}.csv'.format(structured_collection_date))
note_icd9_path = data_path.joinpath('notes_icd9_codes_{}.csv'.format(structured_collection_date))


# In[18]:

labels_df = pd.read_csv(labels_path.as_posix())
note_meta_df = pd.read_csv(note_meta_path.as_posix())
note_icd9_df = pd.read_csv(note_icd9_path.as_posix())


# In[19]:

labels_df.head()


# In[20]:

labels_df.shape


# In[21]:

note_meta_df.head()


# In[22]:

note_icd9_df.head()


# # Inspect data

# In[23]:

note_icd9_df.shape


# In[24]:

note_icd9_df.query('level == "source"').shape


# In[25]:

grouped = note_icd9_df.query('level == "source"').groupby('code').count().sort_values('md5', ascending=False)['md5']
display.display(grouped.head())
print(grouped.shape)


# In[26]:

grouped = note_icd9_df.query('level == "0"').groupby('code').count().sort_values('md5', ascending=False)['md5']
display.display(grouped.head())
print(grouped.shape)


# In[27]:

grouped = note_icd9_df.query('level == "1"').groupby('code').count().sort_values('md5', ascending=False)['md5']
display.display(grouped.head())
print(grouped.shape)


# In[28]:

grouped = note_icd9_df.query('level == "2"').groupby('code').count().sort_values('md5', ascending=False)['md5']
display.display(grouped.head())
print(grouped.shape)


# In[29]:

grouped = note_icd9_df.query('level == "3"').groupby('code').count().sort_values('md5', ascending=False)['md5']
display.display(grouped.head())
print(grouped.shape)


# In[30]:

grouped = note_icd9_df.query('level == "4"').groupby('code').count().sort_values('md5', ascending=False)['md5']
display.display(grouped.head())
print(grouped.shape)


# From above, see that there are no "level 4" codes.  As we increase from level 0 to level 3 we get more specific codes, with corresponding increase in number of codes and decrease in the maximum frequency of occurence.  

# # Assemble data for classification

# ## As a first pass, start with a single diagnosis level, combine with labels, inspect

# In[27]:

icd9_1lev = note_icd9_df.query('level == "1"')
icd9_1lev.head()


# In[28]:

labels_df.head()


# In[29]:

tmp_df = icd9_1lev.groupby(['subject_id', 'md5', 'code']).agg({'level': lambda x: 1})
tmp_df.rename(columns={'level': 'code'}, inplace=True)
icd9_vec_df = tmp_df.unstack(fill_value=0)
#icd9_vec_df.columns = icd9_vec_df.columns.droplevel()


# In[30]:

icd9_vec_df.head()


# In[125]:

feat_vecs = labels_df.groupby(['subject_id', 'md5']).max()[['category',] + categories]
feat_vecs = feat_vecs.join(icd9_vec_df)


# In[126]:

feat_vecs.head()


# In[132]:

feat_vecs.shape


# In[127]:

feature_cols = [c for c in feat_vecs.columns if isinstance(c, tuple)]


# In[128]:

code_lookup = [{'icd9':icd, 'descr': tree.find(icd[1]).description} for icd in feature_cols]


# In[129]:

code_lookup_df = pd.DataFrame(code_lookup).set_index('icd9')


# Note: not really an odds ratio, dividing by population mean rather than mean of non-flagged population, otherwise many divide by zeros

# In[130]:

all_vecs = feat_vecs[feature_cols].mean()
likely_concepts = dict()
for cat in categories:
    with_label = feat_vecs.loc[feat_vecs[cat]==1, feature_cols].mean()
    with_label = with_label/all_vecs
#     no_label = feat_vecs.loc[feat_vecs[cat]==0, feature_cols].mean()
#     with_label = with_label/no_label
    with_label.name = 'OR'
    with_label = code_lookup_df.join(pd.DataFrame(with_label))
    likely_concepts[cat] = with_label.sort_values('OR', ascending=False)


# In[131]:

for cat in categories[:15]:
    print(cat)    
    display.display(likely_concepts[cat].head(20))


# In[133]:

fit_dat = feat_vecs.dropna().copy()
fit_dat.loc[:, 'random'] = np.random.rand(fit_dat.shape[0], 1)
fit_dat.head()


# In[134]:

test_frac = 0.3
X_test = fit_dat.loc[fit_dat['random'] < test_frac, feature_cols].values
X_train = fit_dat.loc[fit_dat['random'] >= test_frac, feature_cols].values


# In[218]:

classifiers = {}
for cat in categories:
    print(cat)
    Y_test = fit_dat.loc[fit_dat['random'] < test_frac, cat].values
    Y_train = fit_dat.loc[fit_dat['random'] >= test_frac, cat].values    

    logreg = linear_model.LogisticRegressionCV() #class_weight={0: .05, 1: .95})
    logreg.fit(X_train, Y_train)    

    ranked_df = pd.DataFrame([{'icd9':i[0], 'weight': i[1]} for i in zip(feature_cols, logreg.coef_[0,:])]).      set_index('icd9')    
    ranked_df = code_lookup_df.join(ranked_df).sort_values('weight', ascending=False)
    display.display(ranked_df.head(10))

    Y_pred = logreg.predict_proba(X_test)[:, 1]
    
    [fpr, tpr, thresh] = metrics.roc_curve(Y_test, Y_pred)
    auc = metrics.auc(fpr, tpr)
    thresh_ind = np.abs(tpr-0.5).argmin()

    plt.plot(fpr, tpr)
    plt.plot(fpr[thresh_ind], tpr[thresh_ind], marker='.', markersize=10)
    plt.plot([0, 1],[0, 1],'k--')
    plt.grid(True)
    plt.axes().set_aspect('equal') 
    plt.title(cat)    
    plt.xlabel('Specificity (1-FPR)')
    plt.ylabel('Sensitivity (TPR)')

    fig_path = pl.Path(path_config['results_dir']).joinpath('{}_{}_log_reg_roc.png'.format(time_str, cat))
    print('Saving figure to {}'.format(fig_path))
    plt.savefig(fig_path.as_posix())
    
    plt.show()

    print('AUC = {}'.format(auc))
    print('0.5 Sensitivity Probability Threshold = {}'.format(thresh[thresh_ind]))    
    
    print('Confusion matrix:  [TN FP; FN, TP]')
    print(metrics.confusion_matrix(Y_test, Y_pred > thresh[thresh_ind]))
    print('----------------------------------')
    
    classifiers[cat] = {
        'classifier': logreg,
        'threshold': thresh[thresh_ind]
    }


# In[202]:

list(path_config.keys())


# In[213]:

class_dat = {
    'classifiers': classifiers,
    'features': feature_cols
}
clf_path = pl.Path(path_config['results_dir']).joinpath('{}_icd9_log_reg.pkl'.format(time_str))
print('Saving classifiers to {}'.format(clf_path))
with open(clf_path.as_posix(), 'wb') as f:
    pkl.dump(class_dat, f)


# In[ ]:



