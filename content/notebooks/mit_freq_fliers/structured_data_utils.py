import logging
import pandas as pd
import sys 

sys.path.append('./icd9/')
from icd9 import ICD9

# feel free to replace with your path to the json file
tree = ICD9('icd9/codes.json')

logger = logging.getLogger()


def get_note_metadata(conn, row_id):
    """Retrieve note metadata from MIMIC III database

    Parameters
    ----------
    conn : sqlalchemy connection
    row_id : MIMIC III note row id to retrieve

    Returns
    -------
    dict : subject_id, hadm_id, chartdate, charttime, storetime, cgid corresponding to note
    """

    query = """
select subject_id, hadm_id, chartdate, charttime, storetime, cgid
from mimiciii.noteevents
where row_id={};"""
    
    res = conn.execute(query.format(int(row_id))).fetchone()
    
    if res is None:
        return None

    return dict(res)


def clean_icd9_code(icd9_str):
    """Convert a MIMIC III-style ICD9 code to a standard code for lookup

    Parameters
    ----------
    icd9_str : str
        MIMIC III code (e.g. '39891')

    Returns
    -------
    str :
        Standard ICD9 code, e.g. 398.91
    """
    if icd9_str is None:
        return None
    
    if '.' not in icd9_str:
        if icd9_str.startswith('E') and len(icd9_str) > 4:
            icd9_str = icd9_str[:4] + '.' + icd9_str[4:]
        elif len(icd9_str) > 3:
            icd9_str = icd9_str[:3] + '.' + icd9_str[3:]
        
    return icd9_str


def print_icd9_tree(node):
    """Print the ICD9 tree of a given ICD9 code

    Parameters
    ----------
    node : str
        Properly formatted ICD9 code (e.g. '398.91')

    """
    if isinstance(node, str):
        icd9_str = clean_icd9_code(node)
        node = tree.find(icd9_str)
    
    if node is not None:    
        print('Parents:')
        for c in node.parents:
            print('- {}: {}'.format(c.code, c.description))    

        print('\n-> {}: {}\n'.format(node.code, node.description))

        print('Children:')
        for c in node.children:
            print('- {}: {}'.format(c.code, c.description))


def get_hadm_diagnoses(conn, hadm_id):
    """Retrieve all ICD9 diagnoses for a given encounter ID

    Parameters
    ----------
    conn : sqlalchemy connection
    hadm_id : int or str
        MIMIC III encounter ID

    Returns
    -------
    list of dict : each dict contains the ICD9 code, short title, and long title from MIMIC III

    """
    if hadm_id is None:
        return None
    
    query = """
select a.subject_id, a.hadm_id, a.seq_num, a.icd9_code, diags.short_title, diags.long_title
from mimiciii.diagnoses_icd as a
left join mimiciii.d_icd_diagnoses as diags on a.icd9_code = diags.icd9_code
where a.hadm_id = {}
order by a.seq_num
"""
    res = conn.execute(query.format(int(hadm_id))).fetchall()
    
    if res is not None:
        res = [dict(r.items()) for r in res]
        for r in res:
            r['clean_icd9_code'] = clean_icd9_code(r['icd9_code'])
            r['known_icd9_code'] = tree.find(r['clean_icd9_code']) is not None                
    
    return res


def get_icd9_levels(icd9_code, max_depth=5):
    """Retrieve parents in the ICD9 hierarchy of the given code

    Parameters
    ----------
    icd9_code : str
        Properly formated ICD9 code
    max_depth : int
        Maximum depth to retrieve

    Returns
    -------
    list
        Parents of the given ICD9 code, starting from top-most parent and decending down to max_depth
    """
    icd9_str = clean_icd9_code(icd9_code)
    node = tree.find(icd9_str)
    
    levels = None
    
    if node is not None:
        levels = [c.code for c in node.parents[1:max_depth]]

    return levels
