# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:06:45 2023

@author: surma
"""

import pandas as pd
from dot_makers import all_stunt_info, build_a_gif
import os

os.chdir('../..')
proj_path = os.getcwd()
stunt_path = os.path.join(proj_path, 'dots/stunts')
if not os.path.exists(stunt_path):
    os.makedirs('dots/stunts')
for index, row in all_stunt_info.iloc[:5][['gameId', 'playId', 'stuntId']].iterrows():
    try:
        build_a_gif(row['gameId'], row['playId'], row['stuntId'])
    except Exception as e:
        print('Error in game {} play {} stunt {}'.format(row['gameId'], row['playId'], row['stuntId']))
        print(e)