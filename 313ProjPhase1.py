import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#loading in data
size = 'small'
dp_pairs = pd.read_csv("Data/BS_DP_"+size+".csv")
pl_full = pd.read_csv("Data/BS_PL_"+size+".csv")
mun_reqs = pd.read_csv("Data/BS_MUN_"+size+".csv")

pl_full.rename(columns={'Unnamed: 0':'pls'}, inplace=True)
#speed parameters
walk_speed=3
bike_speed=15

# creating list of all DP pairs
dp_melt = dp_pairs.drop(['LAT', 'LON'], axis=1)
dp_melt = pd.melt(dp_melt,id_vars=['Unnamed: 0'])

# DP pairs that are more than 45 minutes walking
dp_list = dp_melt.loc[dp_melt['value'] >= .75*walk_speed]

pl_vals = pl_full[['pls']]

# add medium bike station at every PL
pl_vals.insert(len(pl_vals.columns), 'vals', 'medium')
pl_vals

# function takes in DP paiir list reduced, pl list, outputs nrew data frame w/ dp pairs and 2 pls for each forming greedy pathself.
# loops through each dp pairs
