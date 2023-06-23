import pandas as pd 
import os

data_dir = "../../data"
games = pd.read_csv(os.path.join(data_dir, "games.csv"))
pffScoutingData = pd.read_csv(os.path.join(data_dir, "pffScoutingData.csv"))
players = pd.read_csv(os.path.join(data_dir, "players.csv"))
plays = pd.read_csv(os.path.join(data_dir, "plays.csv"))
tracking_files = [fname for fname in os.listdir(data_dir) if fname.startswith('week')]
weeks = [pd.read_csv(os.path.join(data_dir, fname)) for fname in tracking_files]

for i, week in enumerate(weeks):
    week['week'] = i+1

all_weeks = pd.concat(weeks)

# filtering out designed roll-outs
dropBacks = plays[(plays['dropBackType'].isin(['TRADITIONAL','SCRAMBLE'])) &
                  (plays['pff_playAction'] == False)][['gameId','playId']]
# merge dropback passes with play scouting data
dropBackScoutingData = dropBacks.merge(pffScoutingData, on = ['gameId','playId'])
# merge that with tracking data
all_players = all_weeks.merge(dropBackScoutingData, on = ['gameId','playId','nflId'])

# get the ball location for each frame
ball_at_snap = all_weeks[(all_weeks.team == 'football') & (all_weeks.event == 'ball_snap')]
ball_at_snap = ball_at_snap[['gameId','playId','y']]
ball_at_snap = ball_at_snap.rename(columns={"y": "ball_y_snap"})
# x coordinate (yard line) for tracking data is unreliable, using the charted yard line
abs_x = plays[['gameId','playId','absoluteYardlineNumber']].copy()
abs_x.columns.values[2] = 'ball_x_snap'
ball_at_snap = ball_at_snap.merge(abs_x,on = ['gameId','playId'])

# merge ball location with player data
all_players_w_ball = all_players.merge(ball_at_snap, on = ['gameId','playId'])


