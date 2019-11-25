from sklearn.utils import resample
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def df_resample(df, column, function=np.max):
    counts = df[column].value_counts()
    size = function(counts)
    final_df = pd.DataFrame()
    for x in counts.keys():
        ds_upsampled = resample(df[df[column] == x],
                                replace=True,  # sample with replacement
                                n_samples=size,  # to match majority class
                                random_state=123)  # reproducible results

        final_df = pd.concat([final_df, ds_upsampled])

    return final_df


def df_upsample(df, column):
    return df_resample(df, column)


def df_downsample(df, column):
    return df_resample(df, column, np.min)


def drop_multicollinearity(df, show=False):
    dummy_list = [['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5',
                   'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9',
                   'parentesco10', 'parentesco11', 'parentesco12'],
                  ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc',
                   'paredfibras', 'paredother'],
                  ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene',
                   'pisomadera'],
                  ['techozinc', 'techoentrepiso', 'techocane', 'techootro'],
                  ['abastaguadentro', 'abastaguafuera', 'abastaguano'],
                  ['public', 'planpri', 'noelec', 'coopele'],
                  ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6'],
                  ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4'],
                  ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu6'],
                  ['epared1', 'epared2', 'epared3'],
                  ['etecho1', 'etecho2', 'etecho3'],
                  ['eviv1', 'eviv2', 'eviv3'],
                  ['male', 'female'],
                  ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5',
                   'estadocivil6', 'estadocivil7'],
                  ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
                   'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'],
                  ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5'],
                  ['region_central', 'region_chorotega', 'region_pacifico_central', 'region_brunca',
                   'region_huetar_atlantica', 'region_huetar_norte'],
                  ['zona_urbana', 'zona_rural']
                  ]

    drop_list = []
    for dummy in dummy_list:

        k = dummy[0]
        colin = pd.DataFrame(data=df[k], columns=[k])
        colin['suma'] = df[df[dummy].columns.difference([k])].sum(axis=1)
        corr = colin.corr(method='spearman')
        if show:
            print(k)
            print(corr)
        drop_list.append(dummy[-1])
        # plt.figure(figsize = (8,5))
        # sns.heatmap(corr, annot=True,fmt="f", vmin=-1, vmax=1)

    return df.drop(drop_list, axis=1)


def calculate_vif_(df, thresh=5):
    '''
    Calculates VIF each feature in a pandas dataframe
    A constant must be added to variance_inflation_factor or the results will be incorrect

    :param X: the pandas dataframe
    :param thresh: the max VIF value before the feature is removed from the dataframe
    :return: dataframe with features removed
    '''
    const = add_constant(df)
    cols = const.columns
    variables = np.arange(const.shape[1])
    vif_df = pd.Series([variance_inflation_factor(const.values, i)
                        for i in range(const.shape[1])],
                       index=const.columns).to_frame()

    vif_df = vif_df.sort_values(by=0, ascending=False).rename(columns={0: 'VIF'})
    vif_df = vif_df.drop('const')
    vif_df = vif_df[vif_df['VIF'] > thresh]

    print('Features above VIF threshold:\n')
    print(vif_df[vif_df['VIF'] > thresh])

    col_to_drop = list(vif_df.index)

    for i in col_to_drop:
        print('Dropping: {}'.format(i))
        df = df.drop(columns=i)

    return df
