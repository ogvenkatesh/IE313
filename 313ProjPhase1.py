import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#loading in data
size = 'small'
# dp_pairs = pd.read_csv("C:/Users/sambr/OneDrive/Documents/GitHub/IE313/Data/BS_DP_small.csv")
# pl_full = pd.read_csv("C:/Users/sambr/OneDrive/Documents/GitHub/IE313/Data/BS_PL_small.csv")
# mun_reqs = pd.read_csv("C:/Users/sambr/OneDrive/Documents/GitHub/IE313/Data/BS_MUN_small.csv")

dp_pairs = pd.read_csv("Data/BS_DP_"+size+".csv")
pl_full = pd.read_csv("Data/BS_PL_"+size+".csv")
mun_reqs = pd.read_csv("Data/BS_MUN_"+size+".csv")

pl_full['MUN'] = pl_full['MUN'].astype(int)

pl_full.rename(columns={'Unnamed: 0':'pls'}, inplace=True)
# speed parameters in mph
walk_speed=3
bike_speed=15
#max time in hours
max_time =.75
# sizes of bike stations in number of bikes
small_size = 10
medium_size = 20
large_size = 50
# costs of bike stations in dollars
small_cost = 5000
medium_cost = 8000
large_cost = 20000

#minimum spacing between two nonspecial stations in miles
min_spacing = .25
#maximum distance of station to dp in miles
dp_proximity = 1


# creating list of all DP pairs
dp_melt = dp_pairs.drop(['LAT', 'LON'], axis=1).copy()
dp_melt = pd.melt(dp_melt,id_vars=['Unnamed: 0'])

# DP pairs that are more than 45 minutes walking
dp_list = dp_melt.loc[dp_melt['value'] >= max_time*walk_speed].copy()
dp_list.rename(columns={'Unnamed: 0':'dp_first'}, inplace=True)
dp_list.rename(columns={'variable':'dp_second'}, inplace=True)


# add medium bike station at every PL
pl_vals = pl_full.copy()
# add values e.g. "medium" to end of pl_full
pl_vals.insert(len(pl_vals.columns), 'vals', 'medium')


# mun_counts contains the number of medium bike stations in each mucipality
mun_counts=pl_vals.groupby('MUN')['vals'].apply(lambda x: (x=='medium').sum()).reset_index(name='num_pls').copy()
mun_reqs = mun_reqs.merge(mun_counts, left_on='MUN', right_on='MUN', how='inner').copy()

reduced_pls = pl_vals.loc[(pl_vals['special']==0) & (pl_vals['vals'] != 'none')].reset_index().copy()
violations = (reduced_pls.filter(regex="p\d")<.25).sum(axis=1)
reduced_pls = reduced_pls.assign(violations = violations)
current_pl  = reduced_pls.loc[reduced_pls.index[reduced_pls['violations'].idxmax()], 'pls']

# function takes in DP pair list reduced, pl list, outputs new data frame w/ dp pairs and 2 pls for each forming greedy pathself.
# loops through each dp pairs
def greedy_path(dp_pairs_list, pl_full_matrix):
    pl_full_matrix = pl_full_matrix.loc[pl_full_matrix['vals'] != 'none'].reset_index()
    dp_paths = dp_pairs_list[['dp_first','dp_second']].copy()
    dp_paths = dp_paths.assign(pl_first = np.nan)
    dp_paths = dp_paths.assign(pl_second = np.nan)
    dp_paths = dp_paths.assign(path_time = np.nan)
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

#count = sum(violations)
count=sum(violations)
while count > len(reduced_pls):
    # Check that we will not violate municipality minimum
    current_mun = pl_vals.loc[pl_vals['pls']==current_pl]['MUN'].iloc[0]
    condition = mun_reqs.loc[mun_reqs['MUN']==current_mun]['MIN_BIKES']-(mun_reqs.loc[mun_reqs['MUN']==current_mun]['num_pls']-1)*large_size
    if condition.iloc[0] <=0:
        pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'none'
        # Check that we will not violate 1 mile rule
        if pl_vals.loc[pl_vals['vals'] != 'none'].filter(regex="d\d").min(axis=0).max()<=dp_proximity:
            dp_paths = greedy_path(dp_list, pl_vals)
            # remember to update mun_reqs
            mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']=mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']-1
            # Check that we will not violate 45 minute time limit
            if dp_paths['path_time'].max() > .75:
                #reverting status back to medium
                pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'medium'
                #reverting mun reqs
                mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']=mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']+1
        else:
            pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'medium'
    reduced_pls=pl_vals.loc[(pl_vals['special']==0) & (pl_vals['vals'] != 'none')].reset_index().copy()
    violations = (reduced_pls.filter(regex="p\d")<.25).sum(axis=1)
    reduced_pls=reduced_pls.assign(violations = violations)
    current_pl = reduced_pls.loc[reduced_pls.index[reduced_pls['violations'].idxmax()], 'pls']
    count = sum(violations)


active_pls=pl_vals.loc[pl_vals['vals'] != 'none'].reset_index()

for index, row in active_pls.iterrows(): # replace pl_vals w/ active only
    current_pl = row['pls']
    # Check that we will not violate municipality minimum
    current_mun = pl_vals.loc[pl_vals['pls']==current_pl]['MUN'].iloc[0]
    condition = mun_reqs.loc[mun_reqs['MUN']==current_mun]['MIN_BIKES']-(mun_reqs.loc[mun_reqs['MUN']==current_mun]['num_pls']-1)*large_size
    if condition.iloc[0] <=0:
        pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'none'
        # Check that we will not violate 1 mile rule
        if pl_vals.loc[pl_vals['vals'] != 'none'].filter(regex="d\d").min(axis=0).max()<=dp_proximity:
            dp_paths = greedy_path(dp_list, pl_vals)
            # remember to update mun_reqs
            mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']=mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']-1
            # Check that we will not violate 45 minute time limit
            if dp_paths['path_time'].max() > .75:
                #reverting status back to medium
                pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'medium'
                #reverting mun reqs
                mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']=mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']+1
        else:
            pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'medium'

# change mediums to large and small as necessary for optimal numbers
# This loop changes mediums to larges until minimum is reached or exceeded

for index1, row1 in mun_reqs.iterrows():
    cur_min = mun_reqs.iloc[index1,mun_reqs.columns.get_loc('MIN_BIKES')]
    num_bikes = len(pl_vals.loc[(pl_vals['MUN'] == mun_reqs.loc[index1,'MUN']) & (pl_vals['vals'] == 'medium')])*medium_size
    if num_bikes < cur_min:
        # calculate how many to change to large
        deficit = cur_min - num_bikes
        num_to_change = np.ceil(deficit/(large_size - medium_size))
        # change this many from medium to large_cost
        for index2, row2 in pl_vals.loc[pl_vals['vals'] == 'medium'].iterrows():
            current_pl = 'p'+str(index)
            pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'large'
            num_to_change -= 1
            if num_to_change <= 0:
                break


# Next this loop changes mediums into smalls to be as close to minimum as possible or until all have been changed
for index1, row1 in mun_reqs.iterrows():
    cur_min = mun_reqs.iloc[index1, mun_reqs.columns.get_loc('MIN_BIKES')]
    # THIS NEXT LINE DOESNT LIKE WHEN I CHANGE 'medium' TO 'large'
    num_bikes = len(pl_vals.loc[(pl_vals['MUN'] == mun_reqs.loc[index1,'MUN']) & (pl_vals['vals'] == 'medium')]) + len(pl_vals.loc[(pl_vals['MUN'] == mun_reqs.loc[index1,'MUN']) & (pl_vals['vals'] == 'large')])*large_size
    if num_bikes > cur_min:
        # calculate how many to change to small_cost
        excess = num_bikes - cur_min
        num_to_change = np.floor(excess/(medium_size - small_size))
        # Change this many from medium to small
        for index2, row2 in pl_vals.loc[pl_vals['vals'] == 'medium'].iterrows():
            current_pl = 'p'+str(index)
            pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'small'
            num_to_change -= 1
            if num_to_change <= 0:
                break
