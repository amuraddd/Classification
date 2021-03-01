"""
Load Data
"""
import pandas as pd

def process_a4a(txt_data, no_of_features):
    """
    Convert svm_light a4a data to pandas dataframe
    """
    df_length = txt_data.shape[0] #rows in the dataframe
    
    index = [i for i in range(1, df_length+1)]
    cols = [i for i in range(0, no_of_features+1)]
    df = pd.DataFrame(index=index, columns=cols)
    df2 = df.copy()
    
    for r in range(df_length):
        row = txt_data[0][r].lstrip().rstrip()
        row = row.split(' ')

        for i in range(len(row)):
            popped = row.pop(0)
            if (popped=='-1')|(popped=='+1'):
                df2.iloc[r,[0]] = int(popped)
            elif (popped!='-1')|(popped!='+1'):
                first_element, second_element = popped.split(':')
                df2.iloc[r, int(first_element)] = int(second_element)
    return df2

def process_iris(txt_data, no_of_features):
    """
    Convert svm_light IRIS data to a pandas dataframe
    """
    df_length = txt_data.shape[0] #rows in the dataframe
    
    index = [i for i in range(1, df_length+1)]
    cols = [i for i in range(0, no_of_features+1)]
    df = pd.DataFrame(index=index, columns=cols)
    df2 = df.copy()
    
    for r in range(df_length):
        row = txt_data[0][r].lstrip().rstrip()
        row = row.split(' ')

        for i in range(len(row)):
            popped = row.pop(0)
            if (popped=='1')|(popped=='2')|(popped=='3'):
                df2.iloc[r,[0]] = int(popped)
            elif (popped!='1')|(popped!='2')|(popped!='3'):
                first_element, second_element = popped.split(':')
                df2.iloc[r, int(first_element)] = float(second_element)
    return df2

