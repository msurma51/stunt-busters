# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:47:16 2023

@author: surma
"""

import pandas as pd
import numpy as np
import math
from name_space import *

# transforming the field to put the offense on the bottom and defense on the top w/ ball at the origin
def rel_transformation_xy(row, x = 'x', y = 'y'):
    if (row['playDirection'] == 'right'):
        # 90 deg counterclockwise rotation, translate ball to origin
        y_rel = row[x] - row.ball_x_snap
        x_rel = -1*(row[y] - row.ball_y_snap)
    else:
        # 90 deg clockwise rotation, translate ball to origin
        y_rel = -1*(row[x] - row.ball_x_snap)
        x_rel = row[y] - row.ball_y_snap
    return [x_rel,y_rel]

def rel_transformation_x(row, y = 'y'):
    if (row['playDirection'] == 'right'):
        # 90 deg counterclockwise rotation, translate ball to origin
        return -1*(row[y] - row.ball_y_snap)
    else:
        # 90 deg clockwise rotation, translate ball to origin
        return row[y] - row.ball_y_snap

# rotating each player's orientation to match that above
# offensive players facing ~ 0
def rel_transformation_rot(row):
    if (row['playDirection'] == 'right'):
        # 90 deg counterclockwise rotation
        o_rel = (row.o - 90) % 360
        dir_rel = (row.dir - 90) % 360
    else:
        # 90 deg clockwise rotation
        o_rel = (row.o + 90) % 360
        dir_rel = (row.dir + 90) % 360
    angle_list = [o_rel,dir_rel]
    # set orientation range to (-180,180]
    return [angle - 360 if angle > 180 else angle for angle in angle_list]  

def play_subframe(df,game_id,play_id):
    return df[(df['gameId'] == game_id) & (df['playId'] == play_id)].copy()

def get_alignment(rusher):
    pos, diff, dist, l_surf, r_surf = rusher[['off_position','x_diff','x_dist','l_surface','r_surface']]
    if pos == 'C':
        return '0' if dist < .17 else ('l_1' if diff < 0 else 'r_1')
    if pos == 'LG':
        return 'l_2' if dist < .17 else ('l_2i' if diff > 0 else 'l_3')
    if pos == 'RG':
        return 'r_2' if dist < .17 else ('r_2i' if diff < 0 else 'r_3')
    if pos == 'LT':
        if dist < .17:
            return 'l_4'
        return 'l_4i' if diff > 0 else ('l_5' if l_surf != 'LT' or diff < -1 else 'l_7')
    if pos == 'RT':
        if dist < .17:
            return 'r_4'
        return 'r_4i' if diff < 0 else ('r_5' if l_surf != 'RT' or diff < 1 else 'r_7')
    if pos in ('TE-L', 'TE-iL'):
        if l_surf == pos:
            return 'l_6' if dist < .17 else ('l_6i' if diff > 0 else 'l_7')
        else:
            return 'l_7'
    if pos in ('TE-R', 'TE-iR'):
        if l_surf == pos:
            return 'r_6' if dist < .17 else ('r_6i' if diff < 0 else 'r_7')
        else:
            return 'r_7'
    if pos == 'TE-oL':
        return 'l_7'
    else:
        return 'r_7'

def relative_position(player):
    # Isolate the number associated with player's alignment
    if player['rank'] in (1.0,player['num_rushers']) and player['technique'] > 4:
        return 'E'
    else:
        return 'T' 
    
def slope(row):
    diff_y = row['y2_rel'] - row['y_rel'] 
    diff_x = row['x2_rel'] - row['x_rel']
    if diff_x == 0:
        return np.NaN
    else:
        return diff_y / diff_x
    
def segments_intersect(penetrator_segment, looper_segment):
    # boolean determines whether input line segments intersect
    # intersecting segments can't have the same slope
    if penetrator_segment['m'] != looper_segment['m_looper']:
        # determine x coordinate of intersection point for the lines the respective segments lie on
        # only need x due to 1-to-1 nature of linear functions (duh)
        x = (penetrator_segment['b'] - looper_segment['b_looper']) / (looper_segment['m_looper'] - penetrator_segment['m'])
        # if that x lies in the domain for both segments then the segments also intersect at that x
        in_pen_interval = (min(penetrator_segment['x_rel'],penetrator_segment['x2_rel']) < x 
                           < max(penetrator_segment['x_rel'],penetrator_segment['x2_rel'])) 
        in_looper_interval = (min(looper_segment['x_rel_looper'],looper_segment['x2_rel_looper']) < x 
                           < max(looper_segment['x_rel_looper'],looper_segment['x2_rel_looper'])) 
        if in_pen_interval and in_looper_interval:
            return True
    return False

def ol_exchange(matchup_df):
    # Determine whether an exchange takes place on a stunt by checking whether the matchups after overlap are different than before
    rushers = len(matchup_df['nflId_def'].unique())
    matchups = len(matchup_df[['nflId','nflId_def']].drop_duplicates())
    if rushers == matchups:
        return 0
    else:
        return 1
    

def get_stunt(stunt_keys, game, play, stunt):
    return stunt_keys[(stunt_keys['gameId'] == game) &
                      (stunt_keys['playId'] == play) &
                      (stunt_keys['stuntId'] == stunt)].iloc[0]


# plotting the movements by frame, visual validation check.
def viz(df, game, play, stunt = 0.0):
    #getting rid of the games and plays that are not being looked at
    example = df[(df.gameId == game) & (df.playId == play)]
    for player in example.nflId.unique():
        plt.plot(example[example.nflId == player].x_rel,
                 example[example.nflId == player].y_rel,
                 linestyle="-",
                 marker='.',
                 label=example[example.nflId == player].jerseyNumber.unique())
    plt.title('def: ' + str(example['team'].unique()[0]) + ' --- ' + 'game: ' + str(game) + ' --- ' + 'play: ' + str(play))
    plt.legend()
    plt.show()
    
def position_mapper(pos):
    for chars in ['C','G','TE','HB','FB','WR']:
        if chars in pos:
            return chars
    return 'T'

def def_position_mapper(pos):
    if 'LB' in pos:
        if 'I' in pos:
            return 'ILB'
        if 'O' in pos:
            return 'OLB'
        else:
            return 'LB'
    if 'EO' in pos:
        return 'LEO'
    if 'E' in pos:
        return 'E'
    if 'N' in pos:
        return 'NT'
    else:
        return 'DT'

def qb_triangle_orientation(row):
    # Based on the coordinates of the OL and QB, determine what the orientation of the OL
    # would be if he were facing away from the QB
    # Do this by drawing a triangle with sides:
        # 1. Connecting the OL and QB (hypoteneuse)
        # 2. Vertical line of from the QB's location to the y-coordinate of the OL 
        # 3. Horizontal line from the OL's location to the x-coordinate of the QB
    # Creates 3 angles; one at QB's location, another at OL's location, and a right angle
    # where the horizontal and vertical lines meet
    # We're going to find the measure of the angle at the QB's location
    # First, find the difference in x and y betweeen the OL and QB (okay/helpful to have negative values)
    diff_x = row['x_rel'] - row['qb_x_rel'] #"opposite" segment
    diff_y = row['y_rel'] - row['qb_y_rel'] #"adjacent" segment
    # Find the angle at the QB's location by taking the tan-1 
    if diff_y != 0: # QB and OL not on the same level 
        theta = math.degrees(math.atan(diff_x / diff_y))
    elif diff_x > 0: # If they are, angle can only be 90 (OL to right) or -90 (OL to left)
        return 90.0
    else:
        return -90.0
    if diff_y > 0: #QB deeper than OL
        return theta
    else: # OL deeper than QB
        if diff_x > 0: # theta will return negative, we want 180 + that negative angle 
            return 180 + theta
        elif diff_x < 0: # theta will return position, we want -180 + that positive angle
            return theta - 180
        else: # OL behind the QB
            return 180
    
