import pandas as pd
import numpy as np

class MissingData(): # better name? just DataInstance?
    '''
    Class for the missing data structure 
    '''
    def __init__(self, dataset):
        # maybe just have constructor to initialise it such that no data are missing
        # use dictionary with pairs of tuple of missing field combinations and number
        self.missing_combinations, self.lookup_table_forward = self.compute_missingness(dataset)
        self.missing_fields = self.count_missing_fields(dataset) # only fields? since there's already Missing in the class name?
        self.lookup_table_backward = dict( (v,k) for k,v in self.lookup_table_forward.items() )
        # lookup table (a) forward: field to identifier, (b) backward: identifier to field

    def count_missing_fields(self,dataset):
        '''
        return: pd.Dataframe containing the count for each missing field
        '''
        # field_names = list(dataset)
        # field_ids = list(np.arange(len(field_names)))
        dict_of_counts = dataset.isnull().sum().reset_index(name="Count")
        dict_of_counts.rename(columns={"index": "Field"}, inplace = True)
        # replace field name with field id
        dict_of_counts["Field"] = [self.lookup_table_forward[x] for x in dict_of_counts["Field"] ]
        # # drop never missing fields 
        # empty_rows = dict_of_counts[ dict_of_counts["Count"] == 0 ].index
        # dict_of_counts.drop(empty_rows, inplace = True)
        return dict_of_counts

    def compute_missingness(self, dataset):
        '''
        Computes the missingness of the dataset and mapse each field
        to an integer identifier
        :param dataset: pd.DataFrame holding the entire dataset
        :return: a MissingDataInstance (self?) containing the information about the missingness
        '''

        # map field names to integer and create dict
        field_names = list(dataset)
        # field_ids = list(np.arange(len(field_names)))
        field_ids = [x for x in range(len(field_names)) ]
        dict_fields = dict( zip(field_names, field_ids) )

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
            # TODO: not quite right. return name to test
            name = [dict_fields.get(x) for x in name]
            if name:
                name = tuple(name)
                dict_combinations[name] = rows
        return dict_combinations, dict_fields

    