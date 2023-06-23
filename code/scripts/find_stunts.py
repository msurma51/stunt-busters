# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 05:46:07 2023

@author: surma
"""

import numpy as np
import pandas as pd
import math
from pandas.api.types import CategoricalDtype
from load_data import all_players_w_ball
from name_space import *
from function_space import play_subframe, relative_position, rel_transformation_x, get_alignment, slope, segments_intersect

print('Finding stunt candidates...')
pass_rush = pd.read_csv('../../data/pass_rush.csv', index_col = 0)

# Add snap and throw timing to pass rush dataframe
time_to_throw = pass_rush[['gameId','playId','frameId','event']][(pass_rush['event'] == 'ball_snap') | 
                                                                 (pass_rush['event'] =='pass_forward')].drop_duplicates()
time_to_throw['ball_snapped'] = time_to_throw.groupby(['gameId','playId'])['frameId'].shift(1)
time_to_throw = time_to_throw[~time_to_throw['ball_snapped'].isna()]
time_to_throw['ball_snapped'] = time_to_throw['ball_snapped'].astype('int64')
time_to_throw.rename(columns = {'frameId': 'ball_thrown'}, inplace = True)
time_to_throw['time_to_throw'] = time_to_throw['ball_thrown'] - time_to_throw['ball_snapped']
time_to_throw.drop('event', axis = 1, inplace = True)
threshold = int(time_to_throw['time_to_throw'].median())
# threshold is 26 frames for the MEDIAN time to throw 
pass_rush = pass_rush.merge(time_to_throw, on = ['gameId', 'playId'])


# Add column for number of rushers 
pass_rush['num_rushers'] = pass_rush.groupby(['gameId','playId'])['rank'].transform(max)
# Get location data for defensive front 7 and offensive core LOS personnel
defense = pass_rush[(pass_rush['pff_positionLinedUp'].isin(pff_dl+pff_edge+pff_lb)) &
                    (pass_rush['event'] == 'ball_snap')].copy()
offense = all_players_w_ball[(all_players_w_ball['pff_positionLinedUp'].isin(pff_ol+pff_te)) &
                             (all_players_w_ball['event'] == 'ball_snap')].copy()
# filter out any rushers not aligned on the LOS
defense = defense[(defense.y_rel <= 2)]
# transform the location of the offense in order to compare w/ defensive players
offense['x_rel'] = offense.apply(rel_transformation_x, axis=1 ,result_type = 'expand')
# Rename columns to avoid duplicate column names in later merge
offense.rename(columns = {'pff_positionLinedUp': 'off_position', 'x_rel': 'x_rel_off'}, inplace = True)
# For determining alignment, broadcast columns for the left and right side indicating whether there is
# 1. An inside TE aligned on that side
# 2. Just a single TE on that side
# 3. An open surface (no TE, denoted using the OT to that side)
l_sort = CategoricalDtype(categories = ['LT', 'TE-L', 'TE-iL'], ordered = True)
r_sort = CategoricalDtype(categories = ['RT', 'TE-R', 'TE-iR'], ordered = True)
offense['l_edge_players'] = offense['off_position'].astype(l_sort)
offense['r_edge_players'] = offense['off_position'].astype(r_sort)
offense['l_surface'] = offense.groupby(['gameId', 'playId'])['l_edge_players'].transform(max)
offense['r_surface'] = offense.groupby(['gameId', 'playId'])['r_edge_players'].transform(max)
# Get only the columns we need and rename those that would be duplicates after merging
offense = offense[['gameId', 'playId', 'off_position', 'x_rel_off', 'l_surface', 'r_surface']].copy()
tech = defense.merge(offense, on = ['gameId', 'playId'])
# First, find the difference in x_rel of each defensive player from each offensive player
tech['x_diff'] = tech['x_rel'] - tech['x_rel_off'] 
# Then, find the distance and filter by the minimum distances
tech['x_dist'] = tech['x_diff'].abs()
tech['min_dist'] = tech.groupby(['gameId','playId','nflId'])['x_dist'].transform(min)
tech = tech[tech['x_dist'] == tech['min_dist']]
# Use added features and user-defined function to determine each rusher's alignment and technique
tech['alignment'] = tech.apply(get_alignment, axis = 1)
get_tech = lambda a: int(a.split(sep='_')[1][0]) if a != '0' else 0
tech['technique'] = tech['alignment'].map(get_tech)
tech = tech[['gameId', 'playId', 'nflId', 'alignment', 'technique']]
# Merge alignment and technique info into pass rush
# With inner merge, rushers that are not aligned in the box are removed
pass_rush = pass_rush.merge(tech, on = ['gameId', 'playId', 'nflId'])
pass_rush['rel_pos'] = pass_rush.apply(relative_position, axis = 1)
# Add column with # of remaining rushers after filtering
pass_rush['num_rushers_box'] = pass_rush.groupby(['gameId', 'playId', 'frameId'])['nflId'].transform(len)
# Initialize columns to indicate where, whether and how stunts have taken place
pass_rush[['comp_rank','stunt','penetrator']] = np.NaN
pass_rush['rel_pos_comp'] = ''
# Create dataframe of unique plays within each game to loop through
plays_in_games = pass_rush[['gameId', 'playId']].drop_duplicates()
# Initialize list for storing plays in which overlap happens at the same frame for more than 2 players
# These plays will be removed from pass rush after tagging
multi_overlap_plays = []
for index, play in plays_in_games.iterrows():
    try:
        # Get subframe for a specific play
        rushers = play_subframe(pass_rush, play['gameId'], play['playId'])
        # Determine number of rushers remaining after out-of-box rushers were removed
        # Store snap and throw frames for filtering later
        n, snap_frame, throw_frame = rushers.iloc[0][['num_rushers_box','ball_snapped','ball_thrown']]     
        # Initialize variable that dictates how many rushers to the left of a given rusher you will check
        # for rush path overlap
        shift_left = 1
        overlap = True
        # For this play, store the lesser frame of the ball being thrown or 2.6 seconds after the snap
        # We will only consider overlaps up to this frame
        last_frame = min(throw_frame, snap_frame + threshold)
        # There are n-1 possible shifts and nC2 possible pairings, but if there were no overlaps in the 
        # previous shift no overlaps in future shifts are possible
        while shift_left < n and overlap:
            # Determine whether rush paths overlap in this shift. If not, loop will break
            rushers['x_rel_left'] = rushers.groupby(['gameId', 'playId', 'frameId'])['x_rel'].shift(shift_left)
            rushers['comp_rank'] = rushers.groupby(['gameId', 'playId', 'frameId'])['rank'].shift(shift_left)
            rushers['x_left_diff'] = rushers['x_rel'] - rushers['x_rel_left']
            rushers['overlap'] = rushers['x_left_diff'] <= 0
            # Create a subframe of where overlap occurs within the frames of interest
            overlap_df = rushers[(rushers['overlap'] == True) &
                                 (rushers['frameId'].isin(range(snap_frame,last_frame+1)))]
            # If there are no overlaps in this shift, the loop will break for the play
            overlap = len(overlap_df) > 0
            # If there are overlaps: denote where they start, whether a stunt occurs 
            # and determine roles (penetrator / looper)
            # Perform the following steps for each overlap pairing in the shift
            while len(overlap_df) > 0:
                overlap_row = overlap_df.iloc[0]
                overlap_frame = overlap_row['frameId']
                overlap_rank = overlap_row['rank']
                overlap_row_comp = rushers[(rushers['frameId'] == overlap_frame) & 
                                           (rushers['rank'] == overlap_row['comp_rank'])].iloc[0]
                assert np.isnan(pass_rush['comp_rank'].loc[overlap_row.name]), "Multiple overlap found in game {} play {}".format(play['gameId'],play['playId'])
                assert np.isnan(pass_rush['comp_rank'].loc[overlap_row_comp.name]), "Multiple overlap found in game {} play {}".format(play['gameId'],play['playId'])
                # Update pass_rush with comparison ranks at the frame of overlap                    
                pass_rush.at[overlap_row.name,'comp_rank'] = overlap_row['comp_rank']
                pass_rush.at[overlap_row_comp.name,'comp_rank'] = overlap_row['rank']
                # Determine stunt roles
                y_rel_diff = overlap_row['y_rel'] - overlap_row_comp['y_rel']
                if y_rel_diff < 0:
                    penetrator_rank = overlap_row['rank']
                    looper_rank = overlap_row_comp['rank']
                else:
                    penetrator_rank = overlap_row_comp['rank']
                    looper_rank = overlap_row['rank']
                # isolate the path of the penetrator before overlap and that of the looper after overlap
                # path is defined as the series of line segments connecting their coordinates at the corresponding frames
                # this will be used to determine whether these paths intersect, i.e. the looper went where the penetrator was, i.e. a stunt happened
                penetrator_path_before_overlap = rushers[(rushers['rank'] == penetrator_rank) & 
                                                         (rushers['frameId'] <= overlap_frame)][['frameId','rank','x_rel','y_rel']]
                penetrator_path_before_overlap[['x2_rel','y2_rel']] = penetrator_path_before_overlap[['x_rel','y_rel']].shift(-1)
                penetrator_path_before_overlap = penetrator_path_before_overlap[:-1]
                looper_path_after_overlap = rushers[(rushers['rank'] == looper_rank) & 
                                                    (rushers['frameId'] >= overlap_frame -1)][['frameId','rank','x_rel','y_rel']]
                looper_path_after_overlap[['x2_rel','y2_rel']] = looper_path_after_overlap[['x_rel','y_rel']].shift(-1)
                looper_path_after_overlap = looper_path_after_overlap[:-1]
                # for each segment in both paths, add to the frame the slope and y-intercept of the line on which 
                # that segment resides
                for df in (penetrator_path_before_overlap, looper_path_after_overlap):
                    df['m'] = df.apply(slope, axis=1)
                    df['b'] = np.where(df['m'].isna(), np.NaN, df['y_rel'] - df['m']*df['x_rel'])
                looper_path_after_overlap.columns = [col + '_looper' for col in looper_path_after_overlap.columns]
                # Determine whether paths intersect
                is_stunt = False
                path_merge = penetrator_path_before_overlap.merge(looper_path_after_overlap, how='cross')
                euc_dist = lambda row: math.dist((row['x_rel'], row['y_rel']), (row['x_rel_looper'], row['y_rel_looper']))
                path_merge['dist'] = path_merge.apply(euc_dist, axis=1)
                path_merge.sort_values('dist', inplace = True)
                for index, row in path_merge.iterrows():
                    penetrator_segment = penetrator_path_before_overlap[penetrator_path_before_overlap['frameId'] == 
                                                         row['frameId']].iloc[0]
                    looper_segment = looper_path_after_overlap[looper_path_after_overlap['frameId_looper'] == 
                                                               row['frameId_looper']].iloc[0]
                    if segments_intersect(penetrator_segment, looper_segment):
                        is_stunt = True
                        break
                # If they do, denote that a stunt took place and indicate the penetrator (1.0) and 
                # looper (0.0) in the penetrator column
                if is_stunt:
                    pass_rush.at[overlap_row.name,'stunt'] = 1.0
                    pass_rush.at[overlap_row.name,'rel_pos_comp'] = overlap_row_comp['rel_pos']
                    pass_rush.at[overlap_row_comp.name, 'stunt'] = 1.0
                    pass_rush.at[overlap_row_comp.name, 'rel_pos_comp'] = overlap_row['rel_pos']
                    if penetrator_rank == overlap_row['rank']:
                        pass_rush.at[overlap_row.name,'penetrator'] = 1.0
                        pass_rush.at[overlap_row_comp.name,'penetrator'] = 0.0
                    else:
                        pass_rush.at[overlap_row.name,'penetrator'] = 0.0
                        pass_rush.at[overlap_row_comp.name, 'penetrator'] = 1.0  
                else:
                    # if they don't, denote that as well
                    # since there isn't really a looper or penetrator that distinction is inconsequential, so we ignore it
                    pass_rush.at[overlap_row.name, 'stunt'] = 0.0
                    pass_rush.at[overlap_row_comp.name, 'stunt'] = 0.0
                overlap_df = overlap_df[overlap_df['rank'] > overlap_rank]
            shift_left += 1
    except Exception as e:
        multi_overlap_plays.append((play['gameId'],play['playId']))
        print(str(e))
        
mop_df = pd.DataFrame(multi_overlap_plays, columns = ['gameId', 'playId'])
pass_rush = pass_rush[(~pass_rush['gameId'].isin(mop_df['gameId'])) |
                      (~pass_rush['playId'].isin(mop_df['playId']))]
        
pass_rush.to_csv('../../data/pass_rush_tagged.csv')
print('...stunt candidates found and tagged.')
                
                
                
            

            
            
        

        
    
    
    
