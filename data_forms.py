# import numpy as np
import pandas as pd


class TrainingData:

    def __init__(self, dataframes: list) -> None:
        master_data = pd.concat(dataframes)

        # Sort in chronological time
        self.data = master_data.sort_values(by=['tourney_date'], ascending=True)

        self.partitioned_data = {
            'Training': self.data,
            'Testing': None
        }
    
    
    def partition_data(self, test_size: int):

        tourneys = self.data['tourney_id'].unique()

        test_ids = tourneys[-1 - test_size: -1]

        grouped = self.data.groupby('tourney_id')

        test_groups = []
        for test_id in test_ids:
            test_group = grouped.get_group(test_id)
            test_groups.append(test_group)
        
        testing_df = pd.concat(test_groups)
        
        training_df = grouped.filter(lambda x: x.name not in test_ids)
        # training_df = training_groups.obj

        partitions = {
            'Training': training_df,
            'Testing': testing_df
        }

        self.partitioned_data = partitions
        
    