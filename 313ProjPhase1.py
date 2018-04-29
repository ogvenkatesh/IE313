import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#loading in data
size = 'small'
dp_pairs = pd.read_csv("C:/Users/sambr/OneDrive/Documents/GitHub/IE313/Data/BS_DP_small.csv")
pl_full = pd.read_csv("C:/Users/sambr/OneDrive/Documents/GitHub/IE313/Data/BS_PL_small.csv")
mun_reqs = pd.read_csv("C:/Users/sambr/OneDrive/Documents/GitHub/IE313/Data/BS_MUN_small.csv")
# dp_pairs = pd.read_csv("Data/BS_DP_"+size+".csv")
# pl_full = pd.read_csv("Data/BS_PL_"+size+".csv")
# mun_reqs = pd.read_csv("Data/BS_MUN_"+size+".csv")

pl_full.rename(columns={'Unnamed: 0':'pls'}, inplace=True)
#speed parameters
walk_speed=3
bike_speed=15

# creating list of all DP pairs
dp_melt = dp_pairs.drop(['LAT', 'LON'], axis=1)
dp_melt = pd.melt(dp_melt,id_vars=['Unnamed: 0'])

# DP pairs that are more than 45 minutes walking
dp_list = dp_melt.loc[dp_melt['value'] >= .75*walk_speed]
dp_list.rename(columns={'Unnamed: 0':'dp_first'}, inplace=True)
dp_list.rename(columns={'variable':'dp_second'}, inplace=True)

# add medium bike station at every PL
pl_vals = pl_full[['pls']]
pl_vals.insert(len(pl_vals.columns), 'vals', 'medium')
pl_vals

pl_full
pl_full.loc[pl_full.index[pl_full['d40'].idxmin()], 'pls']


# function takes in DP pair list reduced, pl list, outputs new data frame w/ dp pairs and 2 pls for each forming greedy pathself.
# loops through each dp pairs
def greedy_path(dp_pairs_list, pl_full_matrix):
    dp_paths = dp_pairs_list[['dp_first','dp_second']]
    dp_paths=dp_paths.assign(pl_first = np.nan)
    dp_paths=dp_paths.assign(pl_second = np.nan)
    dp_paths=dp_paths.assign(path_time = np.nan)
    for index, row in dp_pairs_list.iterrows():
        dp_f = row['dp_first']
        dp_s = row['dp_second']
        # find minimum distance PL to DP
        pl_f = pl_full_matrix.loc[pl_full_matrix.index[pl_full_matrix[dp_f].idxmin()], 'pls']
        pl_s = pl_full_matrix.loc[pl_full_matrix.index[pl_full_matrix[dp_s].idxmin()], 'pls']
        dp_paths.loc[index, 'pl_first'] = pl_f
        dp_paths.loc[index, 'pl_second'] = pl_s
        # Calculate time for each path
        dp_paths.loc[index, 'path_time'] = pl_full_matrix.loc[(pl_full_matrix['pls'] == pl_f), dp_f].values/walk_speed + pl_full_matrix.loc[(pl_full_matrix['pls'] == pl_f), pl_s].values/bike_speed + pl_full_matrix.loc[(pl_full_matrix['pls'] == pl_s), dp_s].values/walk_speed

    return(dp_paths)
    #

greedy_path(dp_list, pl_full)
