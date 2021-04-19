import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

from typing import List

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('vader_lexicon')

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


def get_sentiment_score(text: str,
                        output_component: str='compound') -> float:
    """
    Use pre-traind Vader sentiment analyzer to calculate 
    sentiment of the text. 
    
    Args:
        text (str) : text to analyze
        output_component (str) : the component, which should be given in the end.
            Can take one of the following
            
    Returns:
        (float) : output_component of the SentimentIntensityAnalyzer().polarity_scores(text)
    """
    sia = SentimentIntensityAnalyzer()
    result = sia.polarity_scores(text)
    if output_component in ['neg', 'neu', 'pos', 'compound']:
        return result[output_component]
    else:
        print("output_component argument must take one of the following values: 'neg', 'neu', 'pos', 'compound'!")


def add_dummy_cols(df: pd.DataFrame, 
                    cat_cols: List, 
                    dummy_na: bool) -> pd.DataFrame:
    '''
    Args:
        df (DataFrame) : pandas dataframe with categorical variables you want to dummy
        cat_cols (List): list of strings that are associated with names of the categorical columns
        dummy_na (bool) : Bool holding whether you want to dummy NA vals of categorical columns or not
    
    Returns:
        df (DataFrame) : a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    df_non_cat = df.drop(columns=cat_cols, errors='ignore')
    dummy_df = {}
    counter = 0
    for col in cat_cols:
        if col in df.columns:
            dummy_df[col] = pd.get_dummies(df[col], dummy_na=dummy_na, prefix=col)
            counter += dummy_df[col].shape[1]
    
    new_cat_df = pd.concat([dummy_df[c] for c in dummy_df], axis=1)
    
    result = pd.concat([df_non_cat, new_cat_df], axis=1)

    return result


def one_hot_enc_for_amenities(df: pd.DataFrame,
                              amenities_col: str='amenities',
                              drop_thershold: float=0):
    """
    Generate new columns, where each column correspond to the amenity given 
    in the list inside amenities_col
    
    Ags:
        df (DataFrame)     : input dataframe with amenities_col
        amenities_co (str) : name of the amenities column, default = 'amenities'
        drop_threshold (float) : float between 0 and 1; all columns, where the share
            of 1's is less then drop_threshold, will be removed. 
        
    Returns:
        new_df (DataFrame) : output dataframe with new columns and dropped amenities_col
    """

    df_copy = df.copy()
    new_cols = pd.get_dummies(df_copy[amenities_col].apply(pd.Series).stack()).sum(level=0)
    stat = new_cols.sum()/len(new_cols)
    cols_to_drop = list(set(list(stat[stat<drop_thershold].index) + ['']))
    new_cols = new_cols.drop(columns=cols_to_drop)
    df_copy = df_copy.drop(columns=[amenities_col])
    new_df = pd.concat([df_copy, new_cols], axis=1)
    
    return new_df


def preprocess_col_name(name: str) -> str:
    """
    preprocess column names to avoid conflicts with LightGBM
    """
    tokenizer = RegexpTokenizer(r'\w+')
    word_list = tokenizer.tokenize(name)
    new_name = ('_').join(word_list)
    return new_name


def calibrate_price_by_month(predicted_price: float,
                             scrape_date: str,
                             calendar_date: str,
                             neighbourhood: str)->float:
    """
    Tune the predicted price accordingly to seasonality data
    
    Args:
        predicted_price (float) : the first guess of the price, which should be modified
            according to seasonality
        scrape_date (str)       : date of listing scraping in %Y-%m-%d format
        calendar_date (str)     : date of interest, to tune price
        neighbourhood           : neighbourhood of the given listing
    
    Returns:
        resulting_price (float) : recalculated price
    
    """
    month_district_stat = pd.read_csv('Boston_Airbnb_data/month_district_price.csv')
    month_of_prediction = datetime.datetime.strptime(scrape_date, '%Y-%m-%d').month
    month_of_calendar = datetime.datetime.strptime(calendar_date, '%Y-%m-%d').month
    
    prediction_reference_tab = (month_district_stat[month_district_stat['month']==month_of_prediction]
                                [['district', 'price']]
                                .rename(columns={'price': 'ref_price'}))
    month_district_stat = month_district_stat.merge(prediction_reference_tab, on='district', how='left')
    month_district_stat['coef'] = month_district_stat['price']/month_district_stat['ref_price']
    
    coef = month_district_stat[ (month_district_stat['district']==neighbourhood) &
                              (month_district_stat['month']==month_of_calendar)]['coef'].values[0]
    resulting_price = predicted_price*coef
    
    return resulting_price