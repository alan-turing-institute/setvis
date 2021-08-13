import pandas as pd
from scipy.sparse import csr_matrix

class MissingData(): # better name? just DataInstance?
    '''
    Class for the missing data structure 
    '''
    def __init__(self, dataset):
        # maybe just have constructor to initialise it such that no data are missing
        # use dictionary with pairs of tuple of missing field combinations and number
        # missing_fields is df at the moment. better dict?
        self.missing_fields = self.count_missing_fields(dataset) # only fields? since there's already Missing in the class name?
        self.missing_combinations = self.compute_missingness(dataset)
        
    def count_missing_fields(self,dataset):
        '''
        return: pd.Dataframe containing the count for each missing field
        '''
        count = dataset.isnull().sum().reset_index(name='Count')
        count.rename(columns={"index": "Field"}, inplace = True)
        # drop never missing fields 
        empty_rows = count[ count['Count'] == 0 ].index
        count.drop(empty_rows, inplace = True)
        return count

    def compute_missingness(self, dataset):
        '''
        Computes the missingness of the dataset
        :param dataset: pd.DataFrame holding the entire dataset
        :return: a MissingDataInstance (self?) containing the information about the missingness
        '''
        dict_combinations = dict()
        data_missingness = dataset.isnull()
        # remove fields (columns) that are never missing 
        data_missingness = data_missingness.loc[:, (data_missingness.sum(axis=0) != 0)]
        data_missingness = data_missingness.astype(int)
        data_missingness = pd.DataFrame(data_missingness.groupby(list(data_missingness)))
      
        # iterate over all combinations 
        for index, combination in data_missingness.iterrows():
            rows = list(combination.iloc[1].index)
            combination = combination.iloc[1].astype(int)
            name = list(combination.loc[:, ~(combination == 0).all()])
            # if name list is not empty
            if name:
                name = tuple(name)
                dict_combinations[name] = rows
            
        return dict_combinations
