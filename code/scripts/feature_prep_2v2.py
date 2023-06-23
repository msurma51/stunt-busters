# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 05:44:24 2023

@author: surma
"""

import pandas as pd
import numpy as np
import os
import math
from function_space import position_mapper, def_position_mapper, qb_triangle_orientation

print('Isolating 2v2 stunts and engineering features...')

data_dir = '../../data'
stunt_info = pd.read_csv(os.path.join(data_dir,'stunt_info.csv'), index_col = 0)
matchups = pd.read_csv(os.path.join(data_dir,'matchups.csv'), index_col = 0)
stunt_frames = pd.read_csv(os.path.join(data_dir,'stunt_frames.csv'), index_col = 0)

# Isolate 2v2 stunts
stunts_2v2 = stunt_info[(stunt_info['num_rushers'] == 2) &
                        (stunt_info['pro_advantage'] == 0)]

matchups_2v2 = matchups.merge(stunts_2v2[['gameId', 'playId', 'stuntId']], 
                              on = ['gameId', 'playId', 'stuntId'])

pro_frames = stunt_frames[stunt_frames['qb_x_rel'] > -100]
pro_frames_2v2 = pro_frames.merge(matchups_2v2[['gameId','playId','nflId', 'stuntId',
                                                'pff_positionLinedUp']].drop_duplicates(),
                                  on = ['gameId', 'playId', 'nflId'])

# Build out features
pro_frames_2v2['position'] = pro_frames_2v2['pff_positionLinedUp'].map(position_mapper)
pro_frames_2v2[['width', 'depth', 'squareness']] = pro_frames_2v2[['x_rel', 'y_rel', 'o_rel']].abs()
pro_frames_2v2['open_outside'] = np.where(pro_frames_2v2['x_rel'] < 0, -1*pro_frames_2v2['o_rel'], 
                                          pro_frames_2v2['o_rel'])
pro_frames_2v2['moving_outside'] = np.where(pro_frames_2v2['x_rel'] < 0, -1*pro_frames_2v2['dir_rel'], 
                                            pro_frames_2v2['dir_rel'])
# Adjust o_rel to get rotation - how far from 0 OL has rotated
# First check for extreme differences in orientation value from the previous frame
pro_frames_2v2['o_shift'] = pro_frames_2v2.groupby(['gameId','playId','nflId','stuntId'])['o_rel'].shift(-1)
pro_frames_2v2['o_diff'] = pro_frames_2v2['o_shift'] - pro_frames_2v2['o_rel']
# Mark the frames where extreme differences occur ("> 180 deg" turn)
# This will tell us when the OL rotated past 180 degrees and from which direction
pro_frames_2v2['right_180'] = pro_frames_2v2['o_diff'] <= -180.0
pro_frames_2v2['left_180'] = pro_frames_2v2['o_diff'] >= 180.0
pro_frames_2v2['180_frame'] = np.where((pro_frames_2v2['right_180'] == True) | (pro_frames_2v2['left_180'] == True), 
                                       pro_frames_2v2['frameId'], np.NaN)
pro_frames_2v2['180_frame1'] = pro_frames_2v2.groupby(['gameId','playId','nflId','stuntId'])['180_frame'].transform(min)
# Initialize rotation as the orientation. Prior to passing 180 deg these are the same
pro_frames_2v2['rotation'] = pro_frames_2v2['o_rel']
# Find all frames where the OL has passed 180 deg and give those frames the same sign as the frame
# coming immediately before passing 180 deg
rot_players = pro_frames_2v2[pro_frames_2v2['180_frame1'] > 0][['gameId', 'playId', 'nflId', 'stuntId']].drop_duplicates()
for index, player in rot_players.iterrows():
    rot_query = "gameId == {} & playId == {} & nflId == {} & stuntId == {}".format(player['gameId'],player['playId'],
                                                                                   player['nflId'],player['stuntId'])
    player_frames = pro_frames_2v2.query(rot_query)
    rot_frames = player_frames[player_frames['frameId'] >= player_frames['180_frame1']]
    oriented_right = rot_frames.iloc[0]['rotation'] > 0
    for dex, frame in rot_frames[1:].iterrows():
        #rot = frame['rotation']
        if oriented_right and frame['rotation'] < 0:
            #rot = frame['rotation'] + 360
            pro_frames_2v2.at[dex,'rotation'] = frame['rotation'] + 360
        if not oriented_right and frame['rotation'] > 0:
            pro_frames_2v2.at[dex,'rotation'] = frame['rotation'] - 360
            #rot = frame['rotation'] - 360
# Change sign of rotation to reflect whether rotation happened inside or outside wrt OL's initial alignment
# Location on 1st frame after snap is used to give a definitive direction for the C ('outside' is whichever side he is
# on in that frame)
# Unlike open_outside, the sign will not change if/when the OL crosses the midline
pro_frames_2v2['ball_snapped'] = pro_frames_2v2.groupby(['gameId', 'playId'])['frameId'].transform(min)
pro_frames_2v2['x_rel1'] = np.where(pro_frames_2v2['frameId'] == pro_frames_2v2['ball_snapped']+1, 
                                    pro_frames_2v2['x_rel'], np.NaN)
pro_frames_2v2['x_rel1'] = pro_frames_2v2.groupby(['gameId', 'playId', 'nflId'])['x_rel1'].transform(max)
pro_frames_2v2['rotation_outside'] = np.where(pro_frames_2v2['x_rel1'] < 0, -1*pro_frames_2v2['rotation'], 
                                              pro_frames_2v2['rotation'])
pro_frames_2v2.drop(['o_shift', 'o_diff', 'right_180', 'left_180', 
                     '180_frame', '180_frame1', 'x_rel1'], axis = 1, inplace = True)
qb_dist = lambda x: math.dist((x['x_rel'], x['y_rel']),(x['qb_x_rel'],x['qb_y_rel']))
pro_frames_2v2['qb_dist'] = pro_frames_2v2.apply(qb_dist, axis = 1)


        
pro_frames_2v2['qb_o'] = pro_frames_2v2.apply(qb_triangle_orientation, axis = 1)
qb_o_rel = pro_frames_2v2['o_rel'] - pro_frames_2v2['qb_o']
pro_frames_2v2['qb_o_rel'] = np.where(qb_o_rel > 180, qb_o_rel - 360,
                                      np.where(qb_o_rel <= -180, qb_o_rel + 180, qb_o_rel))
pro_frames_2v2['qb_squareness'] = pro_frames_2v2['qb_o_rel'].abs()
# Rank protectors based on width inside-out to merge into the same row with and inside and outside OL
pro_frames_2v2['snap_width'] = np.where(pro_frames_2v2['frameId'] == pro_frames_2v2['ball_snapped'],
                                        pro_frames_2v2['width'], np.NaN)
pro_frames_2v2['snap_width'] = pro_frames_2v2.groupby(['gameId', 'playId', 'nflId'])['snap_width'].transform(max)
pro_frames_2v2['width_rank'] = pro_frames_2v2.groupby(['gameId', 'playId', 'frameId','stuntId'])['snap_width'].rank()
# Do the merge, keeping shared columns as is and renaming duplicate columns with _in or _out suffix
pro_frames_2v2 = pro_frames_2v2.merge(stunt_info, on = ['gameId', 'playId', 'stuntId'])
pf22_inside = pro_frames_2v2[pro_frames_2v2['width_rank'] == 1.0].copy()
pf22_outside = pro_frames_2v2[pro_frames_2v2['width_rank'] == 2.0].copy()
shared_columns = list(stunt_info.columns) + ['frameId', 'time', 'week', 'team', 'ball_snapped']
pf22_inside.rename(columns = {col: col +'_in' for col in pro_frames_2v2.columns 
                              if col not in shared_columns}, inplace = True)
pf22_outside.rename(columns = {col: col +'_out' for col in pro_frames_2v2.columns 
                              if col not in shared_columns}, inplace = True)
pro_frames_merge = pd.merge(pf22_inside, pf22_outside, on = shared_columns)
# Produce features that describe the interaction of the OLs
pro_frames_merge['position_combo'] = pro_frames_merge['position_out'] + pro_frames_merge['position_in']
pro_frames_merge['x_diff'] = pro_frames_merge['width_out'] - pro_frames_merge['width_in']
pro_frames_merge['y_diff'] = pro_frames_merge['depth_out'] - pro_frames_merge['depth_in']
pro_frames_merge['dist'] = np.sqrt(pro_frames_merge['x_diff']**2 + pro_frames_merge['y_diff']**2)
pro_frames_merge['min_qb_dist'] = pro_frames_merge[['qb_dist_in', 'qb_dist_out']].min(axis = 1)
# Use individual rotations to create a feature which describes the extent to which OLs
# have rotated toward or away from each other
# Will be calculated right-to-left
pro_frames_merge['outside_right'] = pro_frames_merge['x_rel_out'] > pro_frames_merge['x_rel_in']
pro_frames_merge['rel_rotation'] = np.where(pro_frames_merge['outside_right'] == True,
                                            pro_frames_merge['rotation_out'] - pro_frames_merge['rotation_in'],
                                            pro_frames_merge['rotation_in'] - pro_frames_merge['rotation_out'])
# Merge in info for rushers involved in stunt
rushers = matchups_2v2[['gameId', 'playId', 'stuntId', 'nflId_def','pff_positionLinedUp_def', 
                    'penetrator','technique']].drop_duplicates()
rushers['def_position'] = rushers['pff_positionLinedUp_def'].map(def_position_mapper)
loopers = rushers[rushers['penetrator'] == 0]
penetrators = rushers[rushers['penetrator'] == 1]
looper_frames = loopers.merge(stunt_frames[['gameId','playId','nflId','frameId',
                                            'pff_hit', 'pff_hurry', 'pff_sack']], 
                              left_on = ['gameId','playId','nflId_def'],
                              right_on = ['gameId', 'playId', 'nflId'])
penetrator_frames = penetrators.merge(stunt_frames[['gameId','playId','nflId','frameId','y_rel',
                                                    'pff_hit', 'pff_hurry', 'pff_sack']], 
                              left_on = ['gameId','playId','nflId_def'],
                              right_on = ['gameId', 'playId', 'nflId'])
dup_fields = ['nflId', 'def_position', 'technique','pff_hit', 'pff_hurry', 'pff_sack']
looper_frames.rename(columns = {field: field + '_looper' for field in dup_fields}, inplace = True)
penetrator_frames.rename(columns = {field: field + '_penetrator' for field in dup_fields}, inplace = True)
penetrator_frames.rename(columns = {'y_rel': 'penetrator_depth'}, inplace = True)
rusher_frames = looper_frames.merge(penetrator_frames, on = ['gameId', 'playId', 'stuntId', 'frameId'])
rusher_cols = [col for col in rusher_frames.columns if '_x' not in col and '_y' not in col]
# Find stunts mislabelled as 'TT' which should be either 'ET' or 'TE' 
pro_frames_merge = pro_frames_merge.merge(rusher_frames[rusher_cols], 
                                          on = ['gameId', 'playId', 'stuntId', 'frameId'])
pot_type_errors = pro_frames_merge[(pro_frames_merge['stunt_type'] == 'TT') &
                               (pro_frames_merge['position_combo'] == 'TG')]
# Correct mislabelled stunt type names in pro_frames_merge and stunt_info
for index, row in pot_type_errors.iterrows():
    if row['technique_penetrator'] > 4:
        pro_frames_merge.at[index, 'stunt_type'] = 'ET'
    if row['technique_looper'] > 4:
        pro_frames_merge.at[index, 'stunt_type'] = 'TE'
types_2v2 = pro_frames_merge[['gameId', 'playId', 'stuntId', 'stunt_type']].drop_duplicates()
types_2v2.index = stunts_2v2.index
for index, row in types_2v2.iterrows():
    stunt_info.at[index, 'stunt_type'] = row['stunt_type']                                                                     
# Depth of penetrator is a prominent feature that will be tested and used in the model
pro_frames_merge['penetrator_depth'] = -1*pro_frames_merge['penetrator_depth']
# Harmonic mean squareness, robust to one extreme value
pro_frames_merge['harm_squareness'] = (2*pro_frames_merge['squareness_out']*pro_frames_merge['squareness_in'] / 
                                       (pro_frames_merge['squareness_out'] + pro_frames_merge['squareness_in']))
# Average (arithmetic mean) squareness
pro_frames_merge['mean_squareness'] = (pro_frames_merge['squareness_out'] + pro_frames_merge['squareness_in']) / 2
# Root mean squared of squareness, weighted towards one extreme value
pro_frames_merge['rms_squareness'] = np.sqrt((pro_frames_merge['squareness_out']**2 + 
                                             pro_frames_merge['squareness_in']**2) / 2)
# Max squareness; its all about the one extreme value
pro_frames_merge['max_squareness'] = pro_frames_merge[['squareness_out','squareness_in']].max(axis = 1)
# Same for relation to qb
pro_frames_merge['harm_qb_squareness'] = (2*pro_frames_merge['qb_squareness_out']*pro_frames_merge['qb_squareness_in'] / 
                                       (pro_frames_merge['qb_squareness_out'] + pro_frames_merge['qb_squareness_in']))
pro_frames_merge['mean_qb_squareness'] = (pro_frames_merge['qb_squareness_out'] + pro_frames_merge['qb_squareness_in']) / 2
pro_frames_merge['rms_qb_squareness'] = np.sqrt((pro_frames_merge['qb_squareness_out']**2 + 
                                             pro_frames_merge['qb_squareness_in']**2) / 2)
pro_frames_merge['max_qb_squareness'] = pro_frames_merge[['qb_squareness_out','qb_squareness_in']].max(axis = 1)
# Two ways of standardizing frames / the time at which events take place
pro_frames_merge['frame_from_snap'] = pro_frames_merge['frameId'] - pro_frames_merge['ball_snapped']
pro_frames_merge['frame_from_overlap'] = pro_frames_merge['frameId'] - pro_frames_merge['first_overlap']

pro_frames_merge.to_csv(os.path.join(data_dir,'pro_frames_merge_2v2.csv'))
stunt_info.to_csv(os.path.join(data_dir,'stunt_info.csv'))

print('...feature engineering for 2v2 stunts complete.')



