import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

def convert_price_to_num(data: pd.DataFrame, 
                         price_col_name: str='price',
                         output_col_name: str='clean_price') -> pd.DataFrame:
    """Convert price of string format '$...' into float format
   
    Args:
        data (DataFrame) : input dataframe with the object-type column with price
        price_col_name (str) : the name of column which contains price in string format
        output_col_name (str) : the name of the output column where converted 
            price value should be kept
            
    Returns:
        output_data (DataFrame) : copy of input data with one additional column
            with output_col_name
    
    """
    output_data = data.copy()
    output_data[output_col_name] = output_data[price_col_name].fillna('n/a').apply(lambda x: x.replace('$', ''))
    output_data[output_col_name] = output_data[output_col_name].apply(lambda x: x.replace(',', ''))
    output_data[output_col_name] = output_data[output_col_name].apply(lambda x: float(x) if x!='n/a' else np.nan)
    
    return output_data


def convert_rate_to_num(data: pd.DataFrame, 
                        rate_col_name: str='rate',
                        output_col_name: str='conv_rate') -> pd.DataFrame:
    """Convert rate of string format '...%' into float format
    
    Args:
        data (DataFrame) : input dataframe with the object-type column with rate
        rate_col_name (str) : the name of column which contains rate in string format
        output_col_name (str) : the name of the output column where converted 
            rate value should be kept
            
    Returns:
        output_data (DataFrame) : copy of input data with one additional column
            with output_col_name
    
    """
    output_data = data.copy()
    output_data[output_col_name] = output_data[rate_col_name].fillna('n/a').apply(lambda x: x.replace('%', ''))
    output_data[output_col_name] = output_data[output_col_name].apply(lambda x: x.replace(',', ''))
    output_data[output_col_name] = output_data[output_col_name].apply(lambda x: float(x) if x!='n/a' else np.nan)
    output_data[output_col_name] /= 100
    
    return output_data


def preprocess_amenities(x: str) -> str:
    "Clean string description of amenities"
    prep_x = (x
              .replace('{', '')
              .replace('}', '')
              .replace('"', '')
              .split(','))
    return prep_x


def get_past_date(date_to: str,
                  str_days_ago: str,
                  date_format: str='%Y-%m-%d') -> str:
    """
    Convert string description of time ago into date
    Args:
        date_to (str)       : given date, towards which the past date should be calculated 
            date_to should be given in date_format. The default format is %Y-%m-%d
        str_days_ago (str)  : string description of time ago, given as yesterday, 2 days ago, 
            3 months ago, 2 years ago
        date_format (str)   : format of date_to
        
    Returns:
        date (str)  :  past date
    
    """
    TODAY = datetime.datetime.strptime(date_to, date_format)
    splitted = str_days_ago.split()
    if len(splitted) == 1 and splitted[0].lower() == 'today':
        return str(TODAY.date())
    elif len(splitted) == 1 and splitted[0].lower() == 'yesterday':
        date = TODAY - relativedelta(days=1)
        return str(date.date())
    elif len(splitted) == 1 and splitted[0].lower() == 'never':
        date = TODAY - relativedelta(days=1e4)
        return str(date.date())
    elif splitted[0] == 'a':
        splitted[0] == '1'
    elif splitted[1].lower() in ['hour', 'hours', 'hr', 'hrs', 'h']:
        date = datetime.datetime.now() - relativedelta(hours=int(splitted[0]))
        return str(date.date())
    elif splitted[1].lower() in ['day', 'days', 'd']:
        date = TODAY - relativedelta(days=int(splitted[0]))
        return str(date.date())
    elif splitted[1].lower() in ['wk', 'wks', 'week', 'weeks', 'w']:
        date = TODAY - relativedelta(weeks=int(splitted[0]))
        return str(date.date())
    elif splitted[1].lower() in ['mon', 'mons', 'month', 'months', 'm']:
        date = TODAY - relativedelta(months=int(splitted[0]))
        return str(date.date())
    elif splitted[1].lower() in ['yrs', 'yr', 'years', 'year', 'y']:
        date = TODAY - relativedelta(years=int(splitted[0]))
        return str(date.date())
    else:
        return "Wrong Argument format"