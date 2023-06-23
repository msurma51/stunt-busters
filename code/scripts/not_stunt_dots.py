# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:50:24 2023

@author: surma
"""

import pandas as pd
import numpy as np
from dot_makers import build_a_gif_basic
import os

general_path = os.path.join(os.getcwd(), 'dots/general')
if not os.path.exists(general_path):
    os.makedirs(general_path)
pass_rush = pd.read_csv('../../data/pass_rush_tagged.csv', index_col = 0)
overlaps = pass_rush[pass_rush['stunt'] == 0.0]
for (game, play), frames in overlaps.groupby(['gameId', 'playId']):
    id_pairs = []
    for index, row in frames.iterrows():
        comp_row = frames[frames['rank'] == row['comp_rank']].iloc[0]
        id_pair = {row['nflId'], comp_row['nflId']}
        if id_pair not in id_pairs:
            id_pairs.append(id_pair)
    for i, id_pair in enumerate(id_pairs):
        build_a_gif_basic(game, play, *list(id_pair), pair = i)

stunt_candidates = pass_rush[pass_rush['stunt'] == 1.0].copy()
stunt_candidates = stunt_candidates.query("gameId < 2021091600")
stunt_info = pd.read_csv('../../data/stunt_info.csv', index_col = 0)
for (game, play), frames in stunt_candidates.groupby(['gameId', 'playId']):
    stunt_info_play = stunt_info.query(f"gameId == {game} & playId == {play}")
    if len(stunt_info_play) == 0:
        search_frames = frames
    elif frames['frameId'].max() > stunt_info_play['last_overlap'].max():
        search_frames = frames[frames['frameId'] > stunt_info_play['last_overlap'].max()]
    else:
        continue
    id_pairs = []
    for index, row in search_frames.iterrows():
        comp_row = search_frames[search_frames['rank'] == row['comp_rank']].iloc[0]
        id_pair = {row['nflId'], comp_row['nflId']}
        if id_pair not in id_pairs:
            id_pairs.append(id_pair)
    for i, id_pair in enumerate(id_pairs):
        build_a_gif_basic(game, play, *list(id_pair), pair = i+10)
    
    