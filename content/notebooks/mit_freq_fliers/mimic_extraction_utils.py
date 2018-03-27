import re
import pandas as pd
from difflib import SequenceMatcher

def clean_text(in_text):
    e = re.compile(r"\s|\[\*\*[^\*]+\*\*\]|Admission Date|Discharge Date|Date of Birth")
    out_text = e.sub("", in_text)
    return(out_text)

def fix_date(date_str):
    """Dates in the discharge file are a funky format, standardize
    """
    
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    month_inds = {m: m_ind for (m, m_ind) in zip(months, range(1, len(months)+1))}
    
    m = re.match('(\d{2})-(\w{3})-(\d{4}) (\d\d:\d\d:\d\d)', '10-DEC-2142 00:00:00')
    if len(m.groups()) > 0:
        (day, month, year, time) = m.groups()
        new_str = pd.Timestamp('{}-{}-{} {}'.format(year, month_inds[month], day, time))
    else:
        new_str = date_str
    return new_str

def similar(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return 1-SequenceMatcher(None, a, b).ratio()
    else:
        return 9999