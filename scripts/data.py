import os
import tarfile
import numpy as np
from io import BytesIO
from urllib.request import urlopen

import pandas as pd
import seaborn as sns
from Levenshtein import ratio


def get_files(start=2008, end=2020):
    """
    Downloads all .tar files from github (https://github.com/Jur1cek/gcj-dataset/blob/master/)
    Unzips it to a directory as the original .csv files
    """
    
    years = np.arange(start, end+1)
    
    
    
    for year in years:
        dirpath = os.path.dirname(os.getcwd())
        csvpath = f'raw_data/gcj{year}.csv'
        untarpath = f'raw_data'
        
        fullcsvpath = os.path.join(dirpath,csvpath)
        fulluntarpath = os.path.join(dirpath,untarpath)
        
        if os.path.isfile(fullcsvpath):
            pass
        else:
            urlf = urlopen(f"https://github.com/Jur1cek/gcj-dataset/raw/master/gcj{year}.csv.tar.bz2")
            tarf = tarfile.open(name=None, fileobj=BytesIO(urlf.read()))
            tarf.extractall(fulluntarpath)
            tarf.close()
            
            
def clean_data(df, **kwargs):
    """Removing older code submissions, leaving only the last one available
    Lower index values correspond to more recent solutions"""
    
    # picking and reordering columns for each data frame
    df = df[['year','round','username','task','file','flines','full_path']]
    
    # dropping code submissions that are not the latest ones
    df = df.drop_duplicates(subset=['year', 'round', 'username', 'task'], keep='first')
    df = df.dropna()
       
    # forcing string conversion
    df['task'] = df['task'].str.lstrip("0")
    df['file'] = df['file'].str.lstrip("0").str.lower()
    df['full_path'] = df['full_path'].str.lower()
    df['round'] = df['round'].str.lstrip("0")    


    # finding the language of code
    df.loc[df['year'] == 2020, 'code_lang'] = df['full_path']
    df.loc[df['year'] != 2020, 'code_lang'] = df['file']
    df['code_lang'] = df['code_lang'].str.split(".").str[-1]
    
    # fixing python3 definition
    df.loc[df['code_lang'] == 'python3', 'code_lang'] = "py"
    
    # getting the length of source code
    df['code_len'] = df['flines'].str.len()

    # selecting before appending to csv file      
    df = df.rename(columns={'file':'file_name','flines':'code_source'})
    df = df.drop(columns=['full_path'])
    
    return df


def join_files():
    tar_path = os.path.join(os.path.dirname(os.getcwd()),'raw_data')
    raw_files = os.listdir(tar_path)
    cleaned_path = os.path.join(os.path.dirname(os.getcwd()),'raw_data/cleaned_dataset.csv')

    # dff = pd.DataFrame(columns=['year','round','username','task','file_name','code_source','code_lang', 'code_len'])

    if os.path.isfile(cleaned_path):
        os.remove(cleaned_path)
        print("File already exists - deleting old version")
        pass
    else:
        pass
    
    for csv_file in raw_files:
        if csv_file.endswith('.csv'):
            fullcsvpath = os.path.join(tar_path, csv_file)
            
            # specifying which columns to use here is more efficient than having to drop them afterwards
            df = pd.read_csv(fullcsvpath,
                            low_memory=False, 
                            usecols=['year','round','username','task','file','flines','full_path'],
                            encoding='utf-8',
                            dtype={'year':'int16',
                                    'round':'str',
                                    'username':'str',
                                    'task':'str',
                                    'file':'str',
                                    'flines':'str',
                                    'full_path':'str'}) 
            df = clean_data(df)
            column_names = df.columns
            df = df[['year','round','username','task','file_name','code_source','code_lang','code_len']]
    
            if not os.path.isfile(cleaned_path):
                df.to_csv(cleaned_path, header=column_names, index=False)
            else:  # else it exists so append without writing the header
                df.to_csv(cleaned_path, mode='a', header=False, index=False)
    return print(f"New file created at {cleaned_path}")


def features_preprocessing(df):

    # picking the most relevant coding languages
    languages = ['cpp', 'py', 'java']
    df = df[df['code_lang'].isin(languages)]
    
    # removing too short, too lenghty code - based on most frequent coders
    df = df.query('code_len > 500 and code_len < 15000')
    
    # keeping only developers who participated in more than 4 rounds and 50% fo the competition
    top_authors = df.groupby(['username', 'year'], as_index=False).agg({'round':'nunique', 'task': 'nunique'})
    top_authors = top_authors.sort_values('round', ascending=False)
    top_authors = top_authors[(top_authors['round']>=5) & (top_authors['task']>=12)]
    ta_list = top_authors['username']
    df = df[df['username'].isin(ta_list)]
    
    # sorting columns and rows
    df = df[['username', 'year', 'round','task', 'code_len', 'code_lang', 'code_source']]
    df = df.sort_values(['username','year', 'round','code_len'], ascending=True)

    # shifting rows down
    df['next_username'] = df['username'].shift(-1)
    df['next_year'] = df['year'].shift(-1)
    df['next_round'] = df['round'].shift(-1)
    df['next_code'] = df['code_source'].shift(-1)
    df['next_code_len'] = df['code_len'].shift(-1)
    
    # filling Na values after shift
    df['next_code_len'] = df['next_code_len'].fillna(0)
    df['next_year'] = df['next_year'].fillna(0)
    
    # fixing fields dtypes
    df['next_year'] = df['year'].astype(int)
    df['next_code_len'] = df['next_code_len'].astype(int)
    df['next_code'] = df['next_code'].astype(str)
    df['code_len'] = df['code_len'].astype(int)    
    
    # # cutting original code by the legth of previous code
    df['next_code_cut'] = df.apply(lambda x: x['next_code'][0:x['code_len']] ,axis=1)    

    #calculating distance between strings
    df['string_distance'] = df.apply(lambda x: ratio(x['next_code_cut'], x['code_source'])
                                     if x['next_round'] == x['round']
                                     and x['next_username'] == x['username']
                                     and x['next_year'] == x['year']
                                     else 0.0, axis=1)
    # rearranging fields 
    df = df[df['string_distance']<0.9]
    df = df[['username', 'year', 'round', 'task', 'code_len', 'code_lang', 'code_source', 'next_code_cut', 'string_distance']]
    
    return df