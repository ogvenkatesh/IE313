import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#loading in data
size = 'medium'
# dp_pairs = pd.read_csv("C:/Users/sambr/OneDrive/Documents/GitHub/IE313/Data/BS_DP_small.csv")
# pl_full = pd.read_csv("C:/Users/sambr/OneDrive/Documents/GitHub/IE313/Data/BS_PL_small.csv")
# mun_reqs = pd.read_csv("C:/Users/sambr/OneDrive/Documents/GitHub/IE313/Data/BS_MUN_small.csv")

# pl_full = pd.read_csv("Users/Rohan/Downloads/BS_PL_small.csv")
# dp_pairs = pd.read_csv("Users/Rohan/Downloads/BS_DP_small.csv")
# mun_reqs = pd.read_csv("Users/Rohan/Downloads/BS_MUN_small.csv")

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

# reduced_pls contains only PLs that are not special and currently have a bike station
reduced_pls = pl_vals.loc[(pl_vals['special']==0) & (pl_vals['vals'] != 'none')].reset_index().copy()
pls = ['pls']
pls.extend(reduced_pls['pls'].values)
more_reduced_pls = reduced_pls[pls].copy()
# violations counts the number of times each PL in reduced_pls violates the .25 mile rule
# Each PL will be couted as violating with itself because the distance from a PL to itself is 0.0
violations = (more_reduced_pls.filter(regex="p\d")<.25).sum(axis=1)
reduced_pls = reduced_pls.assign(violations = violations)
# Begin by removing the Pl with the most violations
current_pl  = reduced_pls.loc[reduced_pls.index[reduced_pls['violations'].idxmax()], 'pls']

# function takes in updated DP PL list, outputs new data frame w/ DP pairs and 2 PLs for each forming greedy pathself.
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

dp_paths = greedy_path(dp_list, pl_vals)

# function takes in DP pair list reduced, pl list, outputs new data frame w/ dp pairs and 2 pls for each forming greedy pathself.
# Different from greedy_path(), speedy_path() only recalculates paths that change because of removed PL
def speedy_path(dp_paths_input, pl_full_matrix, current_pl_input):
    pl_full_matrix = pl_full_matrix.loc[pl_full_matrix['vals'] != 'none'].reset_index()
    dp_paths_reduced = dp_paths_input.loc[(dp_paths_input['pl_first'] == current_pl_input) | (dp_paths_input['pl_first'] == current_pl_input)].copy()
    dp_paths = dp_paths_input.copy()
    for index, row in dp_paths_reduced.iterrows():
        dp_f = row['dp_first']
        dp_s = row['dp_second']
        # find minimum distance PL to DP
        pl_f = pl_full_matrix.loc[pl_full_matrix.index[pl_full_matrix[dp_f].idxmin()], 'pls']
        pl_s = pl_full_matrix.loc[pl_full_matrix.index[pl_full_matrix[dp_s].idxmin()], 'pls']
        dp_paths_reduced.loc[index, 'pl_first'] = pl_f
        dp_paths_reduced.loc[index, 'pl_second'] = pl_s
        # Calculate time for each path
        dp_paths_reduced.loc[index, 'path_time'] = pl_full_matrix.loc[(pl_full_matrix['pls'] == pl_f), dp_f].values/walk_speed + pl_full_matrix.loc[(pl_full_matrix['pls'] == pl_f), pl_s].values/bike_speed + pl_full_matrix.loc[(pl_full_matrix['pls'] == pl_s), dp_s].values/walk_speed
    dp_paths.loc[(dp_paths.dp_first.isin(dp_paths_reduced.dp_first)) & (dp_paths.dp_second.isin(dp_paths_reduced.dp_second))] = dp_paths_reduced.copy()
    return(dp_paths)



#count = sum(violations)
count=sum(violations)
n_max = 0
# This loop removes the Pl with the most violations as allowed until no more violations (except for Pl with self)
while count > len(reduced_pls):
    # Check that we will not violate municipality minimum
    current_mun = pl_vals.loc[pl_vals['pls']==current_pl]['MUN'].iloc[0]
    condition = mun_reqs.loc[mun_reqs['MUN']==current_mun]['MIN_BIKES']-(mun_reqs.loc[mun_reqs['MUN']==current_mun]['num_pls']-1)*large_size
    if condition.iloc[0] <=0:
        pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'none'
        # Check that we will not violate 1 mile rule
        if pl_vals.loc[pl_vals['vals'] != 'none'].filter(regex="d\d").min(axis=0).max()<=dp_proximity:
            dp_paths_temp = speedy_path(dp_paths, pl_vals,current_pl)
            # remember to update mun_reqs
            mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']=mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']-1
            # Check that we will not violate 45 minute time limit
            if dp_paths_temp['path_time'].max() > .75:
                #reverting status back to medium
                pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'medium'
                #reverting mun reqs
                mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']=mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']+1
        else:
            pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'medium'
    # if current pl's value stayed medium, now use n+1th maximum
    if pl_vals.loc[pl_vals['pls']==current_pl, 'vals'].iloc[0] == 'medium':
        n_max += 1
    # if it was removed update dp_paths
    if pl_vals.loc[pl_vals['pls']==current_pl, 'vals'].iloc[0] == 'none':
        dp_paths = dp_paths_temp.copy()
    # update the following before returning to top of loop
    reduced_pls = pl_vals.loc[(pl_vals['special']==0) & (pl_vals['vals'] != 'none')].reset_index().copy()
    pls = ['pls']
    pls.extend(reduced_pls['pls'].values)
    more_reduced_pls = reduced_pls[pls].copy()

    violations = (more_reduced_pls.filter(regex="p\d")<.25).sum(axis=1)
    reduced_pls = reduced_pls.assign(violations = violations)
    current_pl = reduced_pls.sort_values('violations', ascending = False, inplace = False).iloc[n_max]['pls']
    count = sum(violations)

# active PLs have a bike station at them
active_pls=pl_vals.loc[pl_vals['vals'] != 'none'].reset_index()

# This loop goes through all active PLs removing more as long as no other constraints will be violated
for index, row in active_pls.iterrows():
    current_pl = row['pls']
    # Check that we will not violate municipality minimum
    current_mun = pl_vals.loc[pl_vals['pls']==current_pl]['MUN'].iloc[0]
    condition = mun_reqs.loc[mun_reqs['MUN']==current_mun]['MIN_BIKES']-(mun_reqs.loc[mun_reqs['MUN']==current_mun]['num_pls']-1)*large_size
    if condition.iloc[0] <=0:
        pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'none'
        # Check that we will not violate 1 mile rule
        if pl_vals.loc[pl_vals['vals'] != 'none'].filter(regex="d\d").min(axis=0).max()<=dp_proximity:
            dp_paths_temp = speedy_path(dp_paths, pl_vals,current_pl)
            # remember to update mun_reqs
            mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']=mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']-1
            # Check that we will not violate 45 minute time limit
            if dp_paths_temp['path_time'].max() > .75:
                #reverting status back to medium
                pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'medium'
                #reverting mun reqs
                mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']=mun_reqs.loc[mun_reqs['MUN']==current_mun,'num_pls']+1
        else:
            pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'medium'
    if pl_vals.loc[pl_vals['pls']==current_pl, 'vals'].iloc[0] == 'none':
        dp_paths = dp_paths_temp.copy()


# Change mediums to large and small as necessary for optimal numbers
# This loop changes mediums to larges until minimum is reached or exceeded
for index1, row1 in mun_reqs.iterrows():
    cur_min = mun_reqs.iloc[index1,mun_reqs.columns.get_loc('MIN_BIKES')]
    num_bikes = len(pl_vals.loc[(pl_vals['MUN'] == mun_reqs.loc[index1,'MUN']) & (pl_vals['vals'] == 'medium')])*medium_size
    if num_bikes < cur_min:
        # calculate how many to change to large
        deficit = float(cur_min - num_bikes)
        num_to_change = np.ceil(deficit/(large_size - medium_size))
        # change this many from medium to large_cost
        pls_to_change = pl_vals.loc[(pl_vals['vals'] == 'medium') & (pl_vals['MUN'] == mun_reqs.loc[index1,'MUN'])]
        for index2, row2 in pls_to_change.iterrows():
            current_pl = 'p'+str(index2)
            pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'large'
            num_to_change -= 1
            if num_to_change <= 0:
                break


# Next this loop changes mediums into smalls to be as close to minimum as possible or until all have been changed
for index1, row1 in mun_reqs.iterrows():
    cur_min = mun_reqs.iloc[index1, mun_reqs.columns.get_loc('MIN_BIKES')]
    num_bikes = len(pl_vals.loc[(pl_vals['MUN'] == mun_reqs.loc[index1,'MUN']) & (pl_vals['vals'] == 'medium')])*medium_size + len(pl_vals.loc[(pl_vals['MUN'] == mun_reqs.loc[index1,'MUN']) & (pl_vals['vals'] == 'large')])*large_size
    if num_bikes > cur_min:
        # calculate how many to change to small_cost
        excess = float(num_bikes - cur_min)
        num_to_change = np.floor(excess/(medium_size - small_size))
        # Change this many from medium to small
        pls_to_change = pl_vals.loc[(pl_vals['vals'] == 'medium') & (pl_vals['MUN'] == mun_reqs.loc[index1,'MUN'])]
        for index2, row2 in pls_to_change.iterrows():
            current_pl = 'p'+str(index2)
            pl_vals.loc[pl_vals['pls']==current_pl, 'vals'] = 'small'
            num_to_change -= 1
            if num_to_change <= 0:
                break




# Feasibility Checker
# municipality min/max
# .25 rule
# 1 mile rule
# 45 minute rule

pl_vals
mun_reqs
def check_sol(pl_vals_input,mun_reqs_input):
    checker = []
    for index1, row1 in mun_reqs.iterrows():
        num_bikes = len(pl_vals_input.loc[(pl_vals_input['MUN'] == mun_reqs_input.loc[index1,'MUN']) & (pl_vals_input['vals'] == 'medium')])*medium_size + len(pl_vals_input.loc[(pl_vals['MUN'] == mun_reqs_input.loc[index1,'MUN']) & (pl_vals_input['vals'] == 'large')])*large_size + len(pl_vals_input.loc[(pl_vals_input['MUN'] == mun_reqs_input.loc[index1,'MUN']) & (pl_vals_input['vals'] == 'small')])*small_size
        cur_min = mun_reqs_input.iloc[index1, mun_reqs_input.columns.get_loc('MIN_BIKES')]
        cur_max = mun_reqs_input.iloc[index1, mun_reqs_input.columns.get_loc('MAX_BIKES')]
        if (num_bikes >= cur_min) & (num_bikes <= cur_max):
            checker.append(0)
        else:
            checker.append(1)
    reduced_pls = pl_vals_input.loc[(pl_vals_input['special']==0) & (pl_vals_input['vals'] != 'none')].reset_index().copy()
    pls = ['pls']
    pls.extend(reduced_pls['pls'].values)
    more_reduced_pls = reduced_pls[pls].copy()
    violations = (more_reduced_pls.filter(regex="p\d")<.25).sum(axis=1)
    if sum(violations) == len(reduced_pls):
        checker.append(0)
    else:
        checker.append(1)
    if pl_vals_input.loc[pl_vals['vals'] != 'none'].filter(regex="d\d").min(axis=0).max()<=dp_proximity:
        checker.append(0)
    else:
        checker.append(1)
    final_paths =  greedy_path(dp_list,pl_vals_input)
    if final_paths['path_time'].max() <= .75:
        checker.append(0)
    else:
        checker.append(1)
    # Cost of solution
    total_cost = len(pl_vals.loc[(pl_vals['MUN'] == mun_reqs.loc[index1,'MUN']) & (pl_vals['vals'] == 'medium')])*medium_cost + len(pl_vals.loc[(pl_vals['MUN'] == mun_reqs.loc[index1,'MUN']) & (pl_vals['vals'] == 'large')])*large_cost + len(pl_vals.loc[(pl_vals['MUN'] == mun_reqs.loc[index1,'MUN']) & (pl_vals['vals'] == 'small')])*small_cost
    check_result=[checker,total_cost,]
    return(check_result)



check_sol(pl_vals,mun_reqs)


# Creating Map - trying out
import gmplot
#define the map starting
gmap = gmplot.GoogleMapPlotter(pl_vals['LAT'], pl_vals['LON'], 13)

#loop through all coordinates and grab lats/lons
lats = []
lons = []
for c in coords:
    gmap_coord = clean_coord(c)
    lats.append(gmap_coord[0])
    lons.append(gmap_coord[1])

#add points to map; so add the PL's and the DP's through this
gmap.scatter(lats, lons, 'red', size=100, marker=False)

#save to map
gmap.draw("reddit map.html")
