# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 06:15:33 2023

@author: surma
"""

import pandas as pd
import numpy as np
import os
import math
from name_space import pff_rb
from function_space import play_subframe, ol_exchange

print('Identifying stunts from candidates...')

data_dir = '../../data'
pass_rush = pd.read_csv(os.path.join(data_dir, 'pass_rush_tagged.csv'), index_col = 0)
pass_pro = pd.read_csv(os.path.join(data_dir,'pass_pro.csv'), index_col = 0)

# Isolate and name all individual stunts
stunts_df = pass_rush[pass_rush['stunt'] == 1.0].copy()
stunts_df['stuntId'] = np.NaN
#stunts_df['stunt_type'] = ''
plays_in_games = stunts_df[['gameId','playId']].drop_duplicates()
for index, play in plays_in_games.iterrows():
    # Get a subframe for all rushers involved in a stunt for that play
    play_df = play_subframe(stunts_df, play['gameId'], play['playId']).sort_values(['frameId'])
    # Set a stunt_id so that multiple stunts occuring on a given play can be considered individually
    stunt_id = 1
    while len(play_df) > 0:
        # Get the row and rank of the first rusher to loop
        looper_row = play_df[play_df['penetrator'] == 0.0].iloc[0]
        looper_rank = looper_row['rank']
        # First stunt includes all overlaps where the first looper is a looper
        stunt_df = play_df[((play_df['rank'] == looper_rank) & (play_df['penetrator'] == 0.0)) | 
                           ((play_df['comp_rank'] == looper_rank) & (play_df['penetrator'] == 1.0))]
        for index, frame in stunt_df.iterrows():
            stunts_df.at[index,'stuntId'] = stunt_id
            #stunts_df.at[index,'stunt_type'] = stunt_type
        # Filter out the overlaps just considered. If frames remaining, process will repeat on a separate stunt
        play_df = play_df[(~play_df['rank'].isin(stunt_df['rank'])) |
                          (~play_df['comp_rank'].isin(stunt_df['comp_rank']))]
        stunt_id += 1
stunts_df = stunts_df[~stunts_df['stuntId'].isna()]
stunts_df.rename(columns = {'frameId': 'overlap_frame'}, inplace = True)
stunts_df['time_to_overlap'] = stunts_df['overlap_frame'] - stunts_df['ball_snapped']


# Isolate individual stunts (distinguishes stunts that occur on the same play but are separate from one another)
stunt_keys = stunts_df[['gameId', 'playId', 'stuntId']].drop_duplicates()
# Initialize dataframes that will be returned with player level data (each matchup is a row) and stunt level data
matchup_cols = ['gameId','playId','stuntId','after_overlap',
                'nflId', 'jerseyNumber', 'pff_positionLinedUp',
                'nflId_def', 'jerseyNumber_def', 'pff_positionLinedUp_def', 
                'rank','alignment','technique','rel_pos', 'penetrator']
matchup_df_all = pd.DataFrame(columns = matchup_cols)
stunt_info_df = pd.DataFrame()
# Get player- and stunt- level data for each stunt
for index, stunt in stunt_keys.iterrows():
    # Get frames for all rushers in this play
    rush_frames = play_subframe(pass_rush, stunt['gameId'], stunt['playId'])
    # Get overlap frames for the stunt
    overlap_frames = stunts_df[(stunts_df['gameId'] == stunt['gameId']) &
                             (stunts_df['playId'] == stunt['playId']) &
                             (stunts_df['stuntId'] == stunt['stuntId'])].sort_values('overlap_frame')
    # Get frames for just rushers involved in the stunt
    stunt_frames = rush_frames[rush_frames['nflId'].isin(overlap_frames['nflId'])]
    # Get all pass pro frames for this play
    pro_frames = play_subframe(pass_pro, stunt['gameId'], stunt['playId'])
    snap_frame = stunt_frames['ball_snapped'].iloc[0]
    # Define the overlap frame of the stunt to be the frame of the final overlap in the stunt
    id_keys = ['gameId','playId','frameId']
    # List keys that would be "duplicated" in a merge of rush and pro frames
    dup_keys = ['nflId','jerseyNumber','pff_positionLinedUp','x_rel','y_rel']
    # Add '_def' suffix to those duplicated keys to clarify identification and reference moving forward
    def_dup_keys = [key+'_def' for key in dup_keys]
    # Get identification and tracking info from rush frames, and rename duplicated keys to have '_def' suffix
    def_track_info = rush_frames[id_keys + dup_keys].copy()
    def_track_info.rename(columns = {tup[0]:tup[1] for tup in zip(dup_keys,def_dup_keys)}, inplace = True)
    # Merge rush and pro frames such that each rusher can be compared to each pass protector
    pro_matchup = pro_frames.merge(def_track_info, on = id_keys)
    # Define a function that returns the distance between a rusher and pass protector at a given frame
    euc_dist = lambda row: math.dist((row['x_rel'], row['y_rel']), (row['x_rel_def'], row['y_rel_def']))
    pro_matchup['dist'] = pro_matchup.apply(euc_dist, axis=1)
    # Get the overlap frames of the stunt in reverse chronological order
    overlap_frame_nums = sorted(list(overlap_frames['overlap_frame'].unique()), reverse = True)
    first_overlap = overlap_frame_nums[-1]
    last_overlap = overlap_frame_nums[0]
    # Split matchup frame dfs at the overlap frame to determine matchups before and after overlap
    pro_matchup_before_overlap = pro_matchup[pro_matchup['frameId'].isin(range(first_overlap-5, first_overlap))]
    pro_matchup_after_overlap = pro_matchup[pro_matchup['frameId'].isin(range(last_overlap,last_overlap+5))]
    # Considering before and after overlap (separately in that order), determine which pass protector matched up with each pass rusher in the stunt
    after_overlap = 0
    # Initialize a dataframe to store the matchups for this stunt
    matchup_df = pd.DataFrame()
    for df in (pro_matchup_before_overlap, pro_matchup_after_overlap):
        # Calculate average distance between each comparison pair and sort lowest to highest
        dist_df = df.groupby(['nflId','nflId_def'])['dist'].mean().reset_index().sort_values(['nflId_def','dist'])
        dist_df['order'] = dist_df.groupby('nflId_def').cumcount()
        # Merge in player info for later use
        matchup_info = pro_matchup[['gameId','playId','nflId','jerseyNumber', 'pff_positionLinedUp',
                                    'nflId_def', 'jerseyNumber_def', 'pff_positionLinedUp_def']].drop_duplicates()
        dist_df = dist_df.merge(matchup_info, on = ['nflId', 'nflId_def'])
        # Define matchups as the pairing which minimizes the mean distance before overlap between each rusher
        # and every pass protector
        matchup_temp = dist_df[dist_df['order'] == 0].copy()
        # If one blocker is "matching up" with multiple rushers, must find the a new pairing such that the 
        # relative distance between rushers and their matchup is minimized
        blockers = matchup_temp['nflId'].value_counts()
        # Initialize a dummy protector nflId in case there are more rushers than 
        # pass protectors
        while (blockers > 1).any():
            # Find the (first) blocker who has multiple matchups
            blocker  = blockers.index[0]
            # If there are multiple rushers the offense can't block, consider the matchups
            # for which a real protector is assigned
            if blocker == 0:
                blockers = blockers.iloc[1:]
                continue
            # Get the rushers the blocker is matched up with
            blocker_df = matchup_temp[matchup_temp['nflId'] == blocker].sort_values('dist')
            # Find the matchups of the rushers in blocker_df with protectors that are not yet accounted for
            open_matchups = dist_df[(~dist_df['nflId'].isin(matchup_temp['nflId'])) &
                                    (dist_df['nflId_def'].isin(blocker_df['nflId_def']))].sort_values(['dist', 'order'])
            if len(open_matchups) > 0:
                top_matchup = open_matchups[:1]
                # If the top open matchup is against a back, whoever is farthest from blocker will get the back
                top_matchup_pos = top_matchup['pff_positionLinedUp'].iloc[0]
                if top_matchup_pos in pff_rb:
                    new_matchup = dist_df[(dist_df['nflId_def'] == blocker_df.iloc[-1]['nflId_def']) &
                                          (dist_df['nflId'] == top_matchup['nflId'].iloc[0])][:1]
                else:
                    new_matchup = top_matchup
            else:
                rusher_row = blocker_df.iloc[-1][['gameId', 'playId', 'nflId_def',
                                                  'jerseyNumber_def', 'pff_positionLinedUp_def']]
                rusher_row['nflId'] = 0
                new_matchup = rusher_row.to_frame().transpose()
            matchup_temp = matchup_temp[~matchup_temp['nflId_def'].isin(new_matchup['nflId_def'])]
            matchup_temp = pd.concat((matchup_temp,new_matchup))
            blockers = matchup_temp['nflId'].value_counts()
        matchup_temp['after_overlap'] = after_overlap
        matchup_df = pd.concat((matchup_df, matchup_temp))    
        after_overlap += 1
    # Now that matchups have been determined, determine whether at the last overlap the wider rusher crosses the
    # face of his matchup. This distinguishes being part of the stunt from an incidental overlap due to wash / pursuit
    for num in overlap_frame_nums:
        # Identify the rusher in the overlap who aligned wider
        rushers = overlap_frames[overlap_frames['overlap_frame'] == num]
        wide_rusher = rushers[rushers['technique'] == rushers['technique'].max()].iloc[0]
        # Determine which OL that rusher was matched up against pre-overlap
        matchup = matchup_df[(matchup_df['nflId_def'] == wide_rusher['nflId']) &
                             (matchup_df['after_overlap'] == 0)].iloc[0]
        # Isolate the frames for the rusher and protector involved in that matchup
        matchup_frames = pro_matchup[(pro_matchup['nflId'] == matchup['nflId']) &
                                     (pro_matchup['nflId_def'] == matchup['nflId_def']) &
                                     (pro_matchup['frameId'] <= num)].copy()
        # Determine whether the rusher gets farther inside than the protector before overlap
        # without rushing past his level
        matchup_frames[['x_dist_ball','x_dist_ball_def']] = matchup_frames[['x_rel','x_rel_def']].abs()
        matchup_frames['x_dist'] = matchup_frames['x_dist_ball_def'] - matchup_frames['x_dist_ball']
        matchup_frames['y_dist'] = matchup_frames['y_rel_def'] - matchup_frames['y_rel']
        cross_face_series = (matchup_frames['x_dist'] < 0) & (matchup_frames['y_dist'] > 0)
        # If there is not a cross-face in the last overlap, remove it and re-start the process 
        # with the new last overlap
        if not cross_face_series.any():
            overlap_frames = overlap_frames[overlap_frames['overlap_frame'] != num]
        # Else break the loop -  finding a cross-face in the final overlap suffices to retain the earlier overlaps
        else:
            break  
    if len(overlap_frames) == 0:
        continue
    matchup_df = matchup_df[matchup_df['nflId_def'].isin(overlap_frames['nflId'])]
    # Identify the rank of the final looper in the stunt
    looper_rank = overlap_frames[overlap_frames['penetrator'] == 0.0]['rank'].iloc[0] 
    # Create a subframe with only frames of other rushers penetrating with that looper
    penetrator_df = overlap_frames[overlap_frames['comp_rank'] == looper_rank].copy()
    # Name lists in reverse chronological order the penetrators that the looper overlaps and
    # concludes with the looper
    penetrator_df.sort_values('overlap_frame', ascending = False, inplace = True)
    rel_pos_list = list(penetrator_df['rel_pos'])
    rel_pos_list.append(penetrator_df.iloc[0]['rel_pos_comp'])
    stunt_type = ''.join(rel_pos_list)
    # Create dataframe row w/ base stunt info to hold stunt-level data and add that data
    stunt_info = stunt.to_frame().transpose()
    stunt_info['stunt_type'] = stunt_type
    stunt_info['num_protectors'] = len(matchup_df[matchup_df['nflId'] != 0]['nflId'].unique())
    # If overlaps are removed due to not passing the 'cross-face' test, the overlap frame must be re-calculated
    stunt_info['first_overlap'] = overlap_frames['overlap_frame'].min()
    stunt_info['last_overlap'] = overlap_frames['overlap_frame'].max()
    stunt_info['time_to_overlap'] = stunt_info['first_overlap'] - snap_frame
    # Returns 1.0 if any blocking matchups after overlap were different than those before overlap
    stunt_info['exchange'] = ol_exchange(matchup_df)
    # Isolates the pass pro frames of blocking involved in protecting this stunt
    pro_frames_matchup = pro_frames[pro_frames['nflId'].isin(matchup_df['nflId'])]
    # If any of these protectors was beaten it is deemed a win for the defense
    result_keys = ['pff_beatenByDefender','pff_hitAllowed','pff_hurryAllowed','pff_sackAllowed']
    pressure_df = pro_frames_matchup[result_keys] > 0.0
    pressure_allowed = pressure_df.any(axis = None)
    stunt_info['rush_win'] = np.where(pressure_allowed,1.0,0.0) 
    # Create a subframe which will (ultimately) contain one row per rusher
    overlap_info = overlap_frames[['nflId','rank','alignment','technique','rel_pos','penetrator']].drop_duplicates().copy()
    # If a rusher has multiple roles (penetrator/looper) in this stunt, drop the one with him looping
    overlap_rusher_counts = overlap_info['nflId'].value_counts()
    while (overlap_rusher_counts > 1).any():
        dup_rusher = overlap_rusher_counts.index[0]
        drop_row = overlap_info[(overlap_info['nflId'] == dup_rusher) &
                                (overlap_info['penetrator'] == 0.0)].iloc[0]
        overlap_info.drop(drop_row.name, inplace = True)
        # Reset the count in case of multiple players with multiple roles
        overlap_rusher_counts = overlap_info['nflId'].value_counts()
    # Merge rusher alignment and role data into matchup_df
    overlap_info.rename(columns = {'nflId':'nflId_def'}, inplace = True)
    matchup_df = matchup_df.merge(overlap_info, on = 'nflId_def')
    matchup_df['stuntId'] = stunt['stuntId']
    stunt_info_df = pd.concat((stunt_info_df, stunt_info)) 
    matchup_df_all = pd.concat((matchup_df_all, matchup_df))[matchup_cols]

stunt_info_df.reset_index(drop = True, inplace = True)
# Indicate how many rushers are involved in the stunt
stunt_info_df['num_rushers'] = stunt_info_df['stunt_type'].str.len()
# Indicate how many more protectors than rushers are involved in blocking the stunt
stunt_info_df['pro_advantage'] = stunt_info_df['num_protectors'] - stunt_info_df['num_rushers']
# Filter out stunts likely to be incidental at the top of a normal rush
# (Exchange rate dips at 2.4 seconds, retains > 90% of stunts)
stunt_info_df = stunt_info_df[stunt_info_df['time_to_overlap'] <= 23]
matchup_df_all = matchup_df_all.merge(stunt_info_df[['gameId', 'playId', 'stuntId']],
                                      on = ['gameId', 'playId', 'stuntId'])
matchup_df_all.reset_index(drop = True, inplace = True)
matchup_df_all.sort_values(by = ['gameId', 'playId', 'stuntId','after_overlap'], 
                           inplace = True)

# Get frames for pass rushers involved in a stunt on a particular play
rusher_keys = matchup_df_all[['gameId','playId','nflId_def']].drop_duplicates()
rush_stunt_frames = pass_rush.merge(rusher_keys, left_on = ['gameId', 'playId', 'nflId'], 
                                   right_on = ['gameId', 'playId', 'nflId_def'])
rush_stunt_frames.drop('nflId_def', axis = 1, inplace = True)
# Get frames for pass protectors involved in blocking a stunt on a particular play
pro_keys = matchup_df_all[['gameId','playId','nflId']].drop_duplicates()
pro_stunt_frames = pass_pro.merge(pro_keys, on = ['gameId','playId','nflId'])
# Merge the rusher frames and protector frames into one dataframe and do some cleaning
all_stunt_frames = pd.concat((rush_stunt_frames,pro_stunt_frames))
# Add 'ball_snapped' data to pass pro rows
all_stunt_frames['ball_snapped'] = all_stunt_frames.groupby(['gameId','playId'])['ball_snapped'].transform(max)
all_stunt_frames = all_stunt_frames[all_stunt_frames['frameId'] >= all_stunt_frames['ball_snapped']]
all_stunt_frames.sort_values(by = ['gameId','playId','nflId','frameId'], inplace = True)
all_stunt_frames.reset_index(drop = True, inplace = True)
all_stunt_frames = all_stunt_frames[['gameId','playId','nflId','frameId','week','time', 'team',
                                     'x_rel','y_rel','o_rel','dir_rel', 's', 'a',
                                     'pff_hit', 'pff_hurry', 'pff_sack','qb_x_rel', 'qb_y_rel']]

stunt_info_df.to_csv(os.path.join(data_dir,'stunt_info.csv'))
matchup_df_all.to_csv(os.path.join(data_dir,'matchups.csv'))
all_stunt_frames.to_csv(os.path.join(data_dir,'stunt_frames.csv'))

print('...stunts identified and named.')



