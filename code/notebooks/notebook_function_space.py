import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_partners(dfhc, feature):
    df = dfhc[(dfhc['feat1'] == feature) |
              (dfhc['feat2'] == feature)]
    partners = set(df['feat1']) | set(df['feat2'])
    return partners

def get_multicollinear_sets(data, features, threshold = 0.65):
    df = data[features]
    corr_matrix = df.corr(numeric_only = True)
    pair_list = []
    for i, col in enumerate(corr_matrix.iloc[:,1:].columns):
        i += 1
        corr_search = corr_matrix[col][:i]
        high_corrs = corr_search[corr_search.abs() >= threshold]
        if len(high_corrs) > 0:
            pairs = list(zip([col]*len(high_corrs),high_corrs.index,high_corrs.values))
            #print(pairs)
            pair_list.extend(pairs)
    dfhc = pd.DataFrame(pair_list, columns = ['feat1','feat2','corr'])
    corr_sets = []
    for j, pair in enumerate(pair_list):
        j+=1
        pair_set = set(pair[:2])
        included_sets = [corr_set for corr_set in corr_sets if pair_set.issubset(corr_set)]
        if len(included_sets) > 0:
            continue
        # df_corr = dfhc[(dfhc['feat1'].isin(pair_set)) |
        #                (dfhc['feat2'].isin(pair_set))]
        poss_corr_set = get_partners(dfhc, pair[0]) & get_partners(dfhc, pair[1])
        if len(poss_corr_set) > 3:
            partner_sets = [get_partners(dfhc, feature) for feature in list(poss_corr_set) if feature not in pair_set]
            corr_set = partner_sets[0].intersection(*partner_sets[1:]) & poss_corr_set
            corr_sets.append(corr_set)
        else:
            corr_sets.append(poss_corr_set)
    return corr_sets

def remove_multicollinear_features(features_in, corr_sets, chosen_features):
    chosen_sets = [corr_set for corr_set in corr_sets if len(corr_set & set(chosen_features)) > 0]
    chosen_set_features = set().union(*chosen_sets)
    removed_features = [feat for feat in chosen_set_features if feat not in chosen_features]
    remaining_sets = [corr_set for corr_set in corr_sets if len(corr_set.difference(set(removed_features))) > 1]
    remaining_features = [feat for feat in features_in if feat not in removed_features]
    return remaining_sets, remaining_features   

def plot_feature_comps(data, features, frame_center, group_by, 
                       units = '', fname = '', set_title = [], pad = 3.0):
    fig, axs = plt.subplots((len(features)+1) // 2,2, sharex = True)
    plt.xticks(np.arange(data[frame_center].min(), data[frame_center].max()+1))
    fig.set_figwidth(12)
    fig.set_figheight(4*((len(features)+1) // 2))
    fig.tight_layout(pad = pad)
    for k, feature in enumerate(features):
        if len(features) <= 2:
            dex = k
        else:
            i = k // 2
            j = k % 2
            dex = i,j
        lines = sns.lineplot(
                ax = axs[dex],
                data = data,
                x = frame_center,
                y = feature,
                hue = group_by)
        title = f'Mean {feature} by Frame'
        if len(set_title) > 0:
            title = set_title[k]
        lines.set(title = title)
        if len(units) > 0:
            y_label = feature + f' ({units})'
            lines.set(ylabel = y_label)
        axs[dex].legend(title = '')
    if len(fname) > 0:
        plt.savefig(fname)
    
    
def compare_by_feature(data, feature_list, frame_center, comp_stat = 'mean', plot_features = 'all',
                       units = '', fname = '', set_title = [], pad = 3.0, min_reps = 20):
    if comp_stat == 'mean':
        f = lambda x: x.mean()
    likely_losses = data[data['wl_likely'] == 'Likely Loss']
    likely_wins = data[data['wl_likely'] == 'Likely Win']
    comp_df = likely_losses.groupby(frame_center)['wl_likely'].count().rename('losses').to_frame()
    comp_df['wins'] = likely_wins.groupby(frame_center)['gameId'].count()
    if type(feature_list) != list:
        feature_list = [feature_list]
    for feature in feature_list:
        comp_df[feature + '_Pwin'] = f(likely_losses.groupby(frame_center)[feature])        
        comp_df[feature + '_Rwin'] = f(likely_wins.groupby(frame_center)[feature])
    if plot_features == 'all':
        features_to_plot = feature_list
    elif len(plot_features) < len(feature_list):
        features_to_plot = [feat for feat in plot_features if feat in feature_list]
    else:
        features_to_plot = []
    comp_df_trunc = comp_df.query(f'losses >= {min_reps} and wins >= {min_reps}')
    data_trunc = data[data[frame_center].isin(comp_df_trunc.index)]
    plot_feature_comps(data_trunc, features_to_plot, frame_center, group_by = 'wl_likely',
                        units = units, fname = fname, set_title = set_title, pad = pad)
    return comp_df