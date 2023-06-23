# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:57:32 2023

@author: surma
"""
import pandas as pd

# position breakdown for coloring
wr = ['WR']
rb = ['RB','FB']
te = ['TE']
qb = ['QB']
ol = ['T','G','C']

db = ['SS','FS','CB','DB']
lb = ['MLB','ILB','LB']
dl = ['DE','NT','DT','OLB']

# pff breakdown for coloring
pff_wr = ['LWR', 'SLoWR', 'SLWR', 'SLiWR', 'SRWR', 'SRiWR', 'SRoWR', 'RWR'] 
pff_rb = ['FB-L', 'HB-L', 'FB', 'HB', 'FB-R', 'HB-R']
pff_te = ['TE-oL', 'TE-L', 'TE-iL', 'TE-iR', 'TE-R', 'TE-oR']
pff_qb = ['QB']
pff_ol = ['LT', 'LG', 'C', 'RG', 'RT']
pff_off = pff_wr+pff_rb+pff_te+pff_qb+pff_ol

pff_db = ['SCBiR', 'SCBoR','SS', 'FSL', 'SSL', 'FS', 'FSR', 'SCBL', 'SSR', 'SCBoL', 'SCBiL', 'LCB','RCB', 'SCBR']
pff_lb = ['LLB', 'LILB', 'MLB', 'RILB', 'RLB']
pff_edge = ['LOLB','ROLB']
pff_dl = ['LEO', 'LE', 'DLT', 'NLT', 'NT', 'NRT', 'DRT', 'RE', 'REO']
pff_def = pff_db+pff_lb+pff_edge+pff_dl


# official team colors
tm_color = {
	'ARI' :   {'Red': '#97233F', 'Black': '#000000', 'Yellow': '#FFB612'},
	'ATL' :   {'Red': '#A71930', 'Black': '#000000', 'Silver': '#A5ACAF'},
	'BAL' :   {'Purple': '#241773', 'Black': '#000000', 'Metallic_Gold': '#9E7C0C', 'Red': '#C60C30'},
	'BUF' :   {'Blue': '#00338D', 'Red': '#C60C30'},
	'CAR' :   {'Carolina_Blue': '#0085CA', 'Black': '#101820', 'Silver': '#BFC0BF'},
	'CHI' :   {'Dark_Navy': '#0B162A', 'Orange': '#C83803'},
	'CIN' :   {'Orange': '#FB4F14', 'Black': '#000000'},
	'CLE' :   {'Dark_Brown': '#311D00', 'Orange': '#FF3C00', 'White': '#FFFFFF'},
	'DAL' :   {'Royal_Blue': '#003594', 'Blue': '#041E42', 'Silver': '#869397', 'Silver_Green': '#7F9695', 'White': '#FFFFFF'},
	'DEN' :   {'Broncos_Orange': '#FB4F14', 'Broncos_Navy': '#002244'},
	'DET' :   {'Honolulu_Blue': '#0076B6', 'Silver': '#B0B7BC', 'Black': '#000000', 'White': '#FFFFFF'},
	'GB'  :   {'Dark_Green': '#203731', 'Gold': '#FFB612'},
	'HOU' :   {'Deep_Steel_Blue': '#03202F', 'Battle_Red': '#A71930'},
	'IND' :   {'Speed_Blue': '#002C5F', 'Gray': '#A2AAAD'},
	'JAX' :   {'Teal': '#006778', 'Black': '#101820', 'Gold': '#D7A22A', 'Dark_Gold': '#9F792C'},
	'KC'  :   {'Red': '#E31837', 'Gold': '#FFB81C'},
	'LV'  :   {'Silver': '#A5ACAF','Black': '#000000'},
	'LAC' :   {'Powder_Blue': '#0080C6', 'Sunshine_Gold': '#FFC20E', 'White': '#FFFFFF'},
	'LA'  :   {'Blue': '#003594', 'Gold': '#FFA300', 'Dark_Gold': '#FF8200', 'Yellow': '#FFD100', 'White': '#FFFFFF'},
	'MIA' :   {'Aqua': '#008E97', 'Orange': '#FC4C02', 'Blue': '#005778'},
	'MIN' :   {'Purple': '#4F2683', 'Gold': '#FFC62F'},
	'NE'  :   {'Nautical_Blue': '#002244', 'Red': '#C60C30', 'Silver': '#B0B7BC'},
	'NO'  :   {'Old_Gold': '#D3BC8D', 'Black': '#101820'},
	'NYG' :   {'Dark_Blue': '#0B2265', 'Red': '#A71930', 'Gray': '#A5ACAF'},
	'NYJ' :   {'Gotham_Green': '#125740', 'Black': '#000000', 'White': '#FFFFFF'},
	'PHI' :   {'Green': '#004C54', 'Silver_Jersey': '#A5ACAF', 'Silver_Helmet': '#ACC0C6', 'Black': '#000000', 'Charcoal': '#565A5C'},
	'PIT' :   {'Gold': '#FFB612', 'Black': '#101820', 'Blue': '#003087', 'Red': '#C60C30', 'Silver': '#A5ACAF'},
	'SF'  :   {'Red': '#AA0000', 'Gold': '#B3995D'},
	'SEA' :   {'Navy': '#002244', 'Green': '#69BE28', 'Gray': '#A5ACAF'},
	'TB'  :   {'Red': '#D50A0A', 'Orange': '#FF7900', 'Black': '#0A0A08', 'Grey': '#B1BABF', 'Pewter': '#34302B'},
	'TEN' :   {'Navy': '#0C2340', 'Blue': '#4B92DB', 'Red': '#C8102E', 'Silver': '#8A8D8F', 'Wolf_Grey': '#A2AAAD', 'Steel_Grey': '#54585A'},
	'WAS' :   {'Burgundy': '#5A1414', 'Gold': '#FFB612'}
}

tm_color_spec = {
	'ARI' :   {'Red': '#97233F', 'Black': '#000000', 'Yellow': '#FFB612'},
	'ATL' :   {'Red': '#A71930', 'Black': '#000000', 'Silver': '#A5ACAF'},
	'BAL' :   {'Purple': '#241773', 'Black': '#000000', 'Metallic_Gold': '#9E7C0C', 'Red': '#C60C30'},
	'BUF' :   {'Blue': '#00338D', 'Red': '#C60C30'},
	'CAR' :   {'Carolina_Blue': '#0085CA', 'Silver': '#BFC0BF', 'Black': '#101820'},
	'CHI' :   {'Dark_Navy': '#0B162A', 'Orange': '#C83803'},
	'CIN' :   {'Orange': '#FB4F14', 'Black': '#000000'},
	'CLE' :   {'Dark_Brown': '#311D00', 'Orange': '#FF3C00', 'White': '#FFFFFF'},
	'DAL' :   {'Royal_Blue': '#003594', 'Silver': '#869397', 'Blue': '#041E42', 'Silver_Green': '#7F9695', 'White': '#FFFFFF'},
	'DEN' :   {'Broncos_Orange': '#FB4F14', 'Broncos_Navy': '#002244'},
	'DET' :   {'Honolulu_Blue': '#0076B6', 'Silver': '#B0B7BC', 'Black': '#000000', 'White': '#FFFFFF'},
	'GB'  :   {'Dark_Green': '#203731', 'Gold': '#FFB612'},
	'HOU' :   {'Deep_Steel_Blue': '#03202F', 'Battle_Red': '#A71930'},
	'IND' :   {'Speed_Blue': '#002C5F', 'Gray': '#A2AAAD'},
	'JAX' :   {'Teal': '#006778', 'Black': '#101820', 'Gold': '#D7A22A', 'Dark_Gold': '#9F792C'},
	'KC'  :   {'Red': '#E31837', 'Gold': '#FFB81C'},
	'LV'  :   {'Silver': '#A5ACAF','Black': '#000000'},
	'LAC' :   {'Powder_Blue': '#0080C6', 'Sunshine_Gold': '#FFC20E', 'White': '#FFFFFF'},
	'LA'  :   {'Blue': '#003594', 'Gold': '#FFA300', 'Dark_Gold': '#FF8200', 'Yellow': '#FFD100', 'White': '#FFFFFF'},
	'MIA' :   {'Aqua': '#008E97', 'Orange': '#FC4C02', 'Blue': '#005778'},
	'MIN' :   {'Purple': '#4F2683', 'Gold': '#FFC62F'},
	'NE'  :   {'Nautical_Blue': '#002244', 'Red': '#C60C30', 'Silver': '#B0B7BC'},
	'NO'  :   {'Old_Gold': '#D3BC8D', 'Black': '#101820'},
	'NYG' :   {'Dark_Blue': '#0B2265', 'Red': '#A71930', 'Gray': '#A5ACAF'},
	'NYJ' :   {'Gotham_Green': '#125740', 'Black': '#000000', 'White': '#FFFFFF'},
	'PHI' :   {'Green': '#004C54', 'Silver_Jersey': '#A5ACAF', 'Silver_Helmet': '#ACC0C6', 'Black': '#000000', 'Charcoal': '#565A5C'},
	'PIT' :   {'Gold': '#FFB612', 'Black': '#101820', 'Blue': '#003087', 'Red': '#C60C30', 'Silver': '#A5ACAF'},
	'SF'  :   {'Red': '#AA0000', 'Gold': '#B3995D'},
	'SEA' :   {'Navy': '#002244', 'Green': '#69BE28', 'Gray': '#A5ACAF'},
	'TB'  :   {'Red': '#D50A0A', 'Black': '#0A0A08', 'Orange': '#FF7900', 'Grey': '#B1BABF', 'Pewter': '#34302B'},
	'TEN' :   {'Navy': '#0C2340', 'Red': '#C8102E', 'Blue': '#4B92DB', 'Silver': '#8A8D8F', 'Wolf_Grey': '#A2AAAD', 'Steel_Grey': '#54585A'},
	'WAS' :   {'Burgundy': '#5A1414', 'Gold': '#FFB612'}
}

player_map = pd.read_csv('../../data/players.csv', index_col = 0)['displayName']