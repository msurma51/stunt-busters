import pandas as pd 
from load_data import plays, all_weeks, all_players, all_players_w_ball
from function_space import rel_transformation_xy, rel_transformation_rot

# isolate pass rushers and protectors using charting data
pass_rush = all_players_w_ball[all_players_w_ball['pff_role'] == 'Pass Rush'].copy()
pass_pro = all_players_w_ball[all_players_w_ball['pff_role'] == 'Pass Block'].copy()

# get qb location data for each frame and merge with pass pro data
qb_locs = all_players[all_players['pff_positionLinedUp'] == 'QB'][['gameId','playId','frameId','x','y']].copy()
qb_locs.rename(columns = {'x': 'qb_x', 'y': 'qb_y'}, inplace = True)
pass_pro = pass_pro.merge(qb_locs, on = ['gameId', 'playId', 'frameId']) 

df_list = [pass_rush, pass_pro]

for i in (0,1):
    df = df_list[i]
    df[['x_rel','y_rel']] = df.apply(rel_transformation_xy, axis=1 ,result_type = 'expand')
    df[['o_rel','dir_rel']] = df.apply(rel_transformation_rot, axis=1 ,result_type = 'expand')
    rank_l2r = df[(df.event == 'ball_snap')].copy()
    rank_l2r['rank'] = rank_l2r.groupby(['gameId','playId'])['x_rel'].rank('first')
    rank_l2r = rank_l2r[['gameId','playId','nflId','rank']]
    df_list[i] = df.merge(rank_l2r, on = ['gameId','playId','nflId'])

pass_rush, pass_pro = df_list

pass_pro[['qb_x_rel', 'qb_y_rel']] = pass_pro.apply(rel_transformation_xy, axis=1 , result_type = 'expand',
                                                    x = 'qb_x', y = 'qb_y')

pass_rush = pass_rush.sort_values(by = ['gameId','playId','rank','frameId'])
pass_pro = pass_pro.sort_values(by = ['gameId','playId','rank','frameId'])

pass_rush.to_csv('../../data/pass_rush.csv')
pass_pro.to_csv('../../data/pass_pro.csv')

