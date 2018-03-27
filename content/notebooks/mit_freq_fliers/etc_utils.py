import pandas as pd

from datetime import timedelta, datetime
import matplotlib.pyplot as plt

import time
import pathlib as pl
import yaml
import os

import logging
logger = logging.getLogger()

def load_config_v2(config_file='config.yaml', 
                   creds_file='credentials.yaml'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    with open(config_file, 'r') as conf_file:
        conf = yaml.load(conf_file)

    defaults = {
        'log_dir': './',
        'input_dir': './',
        'results_dir': './',
        'repo_data_dir': './'
    }
    path_config = dict()
    if 'paths' in conf:
        for k in defaults.keys():
            path_config[k] = os.path.expanduser(conf['paths'].get(k, defaults[k]))
    
    creds = None

    with open(creds_file, 'r') as creds_file:
        creds_dat = yaml.load(creds_file)
        creds = creds_dat.copy()
        
    return time_str, path_config, creds

def load_config(config_file='config.yaml', 
                creds_file='credentials.yaml'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    with open(config_file, 'r') as conf_file:
        conf = yaml.load(conf_file)

    defaults = {
        'log_dir': './',
        'input_dir': './',
        'results_dir': './',
        'repo_data_dir': './'
    }
    path_config = dict()
    if 'paths' in conf:
        for k in defaults.keys():
            path_config[k] = os.path.expanduser(conf['paths'].get(k, defaults[k]))
    
    pg_creds = None

    with open(creds_file, 'r') as creds_file:
        creds_dat = yaml.load(creds_file)

    if 'postgres' in creds_dat:
        pg_creds = creds_dat['postgres']
        
    return time_str, path_config, pg_creds

def configure_logging(logger, work_desc, log_directory=None, time_str=None):
    if time_str is None:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
    
    if log_directory is None:
        log_directory = pl.Path('.')
    
    
    if isinstance(log_directory, str):
        log_directory = pl.Path(log_directory)
    
    logger.setLevel(logging.DEBUG)

    log_file = log_directory.joinpath(time_str + '_' + work_desc.replace(' ', '_') + '.log').as_posix()
    print('Logging to {}'.format(log_file))
    logger.info('Logging to {}'.format(log_file))

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)    