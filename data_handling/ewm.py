import pandas as pd
import numpy as np


def detect_outliers(df, col_name):
    """
    给出现极端值的行打上label，并返回剔除极端行的df
    :param df:
    :param col_name:
    :return:
    """
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3 - q1  # Inter quartile range
    fence_low = q1 - 100 * iqr
    fence_high = q3 + 100 * iqr
    df['label'] = 0
    df['label'][df[col_name] > fence_high] = 1
    df['label'][df[col_name] < fence_low] = 1

    return df['label']


def gp_nor(df, col_name):
    """
    对非比率数据进行分组标准化，后再按该列均值作为标准值同向化
    :param df: 待处理数据表(原始表)，type: dataframe
    :param col_name: 指定列数，type: list，eg.['id', 'time', 'turnoverRatio']
    :return: df
    """
    gp = df.groupby("Scode")
    result = pd.DataFrame()

    for i in df["Scode"].unique().tolist():
        val = gp.get_group(i)[col_name]
        val = 0.998 * (val - val[col_name].min(axis=0)) / (
                val[col_name].max(axis=0) - val[col_name].min(axis=0)) + 0.002
        result = pd.concat([result, val], axis=0, ignore_index=False)

    for col in col_name:
        df[col] = result[col]

    return df


def avr_nor(df, col_name):
    """
    对非比率数据进行标准化，每组均值为标准值
    :param df: 待处理数据表(原始表)，type: dataframe
    :param col_name: 指定列数，type: list，eg.['id', 'time', 'turnoverRatio']
    :return:
    """
    gp = df.groupby("Scode")
    gp_mean = gp[col_name].mean()
    result = pd.DataFrame()

    for i in df["Scode"].unique().tolist():
        val = gp.get_group(i)[col_name]

        val_le = val[val <= gp_mean.loc[i, col_name]]
        val_gt = val[val > gp_mean.loc[i, col_name]]

        val_le = 0.998 * (val_le - val[col_name].min(axis=0)) / (
                gp_mean.loc[i, col_name] - val[col_name].min(axis=0)) + 0.002
        val_gt = 0.998 * (val[col_name].max(axis=0) - val_gt) / (
                val[col_name].max(axis=0) - gp_mean.loc[i, col_name]) + 0.002

        val = val_le.fillna(val_gt)
        result = pd.concat([result, val], axis=0, ignore_index=False)

        # for col in val.columns:
        #     if val[col] <= gp_mean.loc[i, col]:
        #         val[col] = 0.998 * (val[col] - val[col].min(axis=0)) / (gp_mean[col] - val[col].min(axis=0)) + 0.002
        #     else:
        #         val[col] = 0.998 * (val[col].max(axis=0) - val[col]) / (val[col].max(axis=0) - gp_mean[col]) + 0.002

    return result


def minimal(data, cols):
    """
    对极小型（越小越优）指标进行无量纲化
    :param data: 待处理数据表(仅有数据值)，type: array
    :param cols: 指定列数，type: list，eg.[3, 5, 6]
    :return: data
    """
    print("minimal processing...")
    for i in range(0, len(cols)):
        data[:, cols[i]] = 0.998 * (max(data[:, cols[i]]) - data[:, cols[i]]) / (
                max(data[:, cols[i]]) - min(data[:, cols[i]])) + 0.002
        # 0.998与0.002是为了使数值大于0
    return data


def medium(df, data, col_name, lb, ub):
    """
    对适中型（某点/某区间最优）指标进行无量纲化；如果是某点最优，则 a == b
    :param df: 待处理数据表，type: dataframe
    :param data: 待处理数据表(仅有数据值)，type: array
    :param col_name: 指定列，type: str
    :param lb: 区间下界，type: int
    :param ub: 区间上界，type: int
    :return: data
    """
    print("medium processing...")
    val = df.loc[:, col_name].to_frame()  # Series会失去列标，下面两行Bool索引将只显示非空值，难以合并
    val_le = val[val <= lb]
    val_gt = val[val > ub]

    val_le = 0.998 * (val_le - val.min()) / (lb - val.min()) + 0.002
    val_gt = 0.998 * (val.max() - val_gt) / (val.max() - ub) + 0.002

    result = val_le.fillna(val_gt)
    # lb < val <= ub 的值填为1
    result = result.fillna(1)

    col = df.columns.get_loc(col_name) - 2
    data[:, col] = result[col_name].values

    return data


def maximal(data, cols):
    """
    对极大型（越大越优）指标进行无量纲化
    :param data: 待处理数据表(仅有数据值)，type: array
    :param cols: 指定列数，type: list，eg.[3, 5, 6]
    :return:
    """
    print("maximal processing...")
    for i in range(0, len(cols)):
        data[:, cols[i]] = 0.998 * (data[:, cols[i]] - min(data[:, cols[i]])) / (
                max(data[:, cols[i]]) - min(data[:, cols[i]])) + 0.002

    return data


def data_hd(df1):
    # 读取数据（原始数据列从Scode开始）
    print("data is loading...")

    # 提取数据值的列名
    col_list = df1.keys()[2:]
    date = df1['Date'].to_frame()

    # 数据归一化（无量纲+标准化（极差法））
    gp_list = ['Wkcpt', 'Ebit_x', 'Ebit_y', 'Busncycle']
    df2 = gp_nor(df1, gp_list)

    # 剔除极端值
    df1['Label_org'] = 0
    for i in col_list[0:-3]:
        df1['Label_org'] += detect_outliers(df2, '' + i + '')  # ''+i+'' 为了使股票code仍然为str
    # 不知道为啥df1这里多出一列'label'....
    del df1['label']
    df3 = df1[df1['Label_org'] == 0]

    # 使用熵值法处理数据
    Data = df3[col_list].values

    minimal_list = ['Dttart', 'Salcstrt', 'Tsalcstrt', 'Ohexprt', 'Busncycle', 'Violation_freq', 'Penalty', 'Busncycle']
    medium_list = ['Npm', 'Curtrt', 'Wkcpt', 'Aslbrt', 'Equrt', 'Ebit_x', 'Ebit_y']
    maximal_list = [i for i in col_list if i not in minimal_list + medium_list]

    # 得出以上列的索引 (-2是减去前两列"Scode"和"Date")
    minimal_col = [df1.columns.get_loc(i) - 2 for i in minimal_list]
    medium_col = [df1.columns.get_loc(i) - 2 for i in medium_list]
    maximal_col = [df1.columns.get_loc(i) - 2 for i in maximal_list]

    # 极小型处理
    Data = minimal(Data, minimal_col)

    # 适中型处理
    # 1  Npm
    Data = medium(df3, Data, medium_list[0], 0.1, 0.1)
    # 2  Curtrt
    Data = medium(df3, Data, medium_list[1], 1.5, 2)
    # 3  Wkcpt
    Data = medium(df3, Data, medium_list[2], Data[:, medium_col[2]].mean(), Data[:, medium_col[2]].mean())
    # 4  Aslbrt
    Data = medium(df3, Data, medium_list[3], 0.4, 0.7)
    # 5  Equrt
    Data = medium(df3, Data, medium_list[4], 0.7, 2.3)
    # 6  Ebit_x
    Data = medium(df3, Data, medium_list[5], Data[:, medium_col[5]].mean(), Data[:, medium_col[5]].mean())
    # 7  Ebit_y
    Data = medium(df3, Data, medium_list[6], Data[:, medium_col[6]].mean(), Data[:, medium_col[6]].mean())

    # 极大型处理
    Data = maximal(Data, maximal_col)
    #
    Data[np.isnan(Data)] = 0

    # 计算信息熵
    m, n = Data.shape
    p = np.zeros((m, n))
    entropy = np.zeros(n)
    k = 1 / np.log(m)
    for j in range(0, n):
        p[:, j] = Data[:, j] / sum(Data[:, j])
    p[np.isnan(p)] = 0

    for j in range(0, n):
        entropy[j] = -k * sum(p[:, j] * np.log(p[:, j]))
    entropy[np.isnan(entropy)] = 0

    # 计算权重
    weight = (1 - entropy) / sum(1 - entropy)

    # 计算得分
    s1 = np.dot(Data[:, 0:3], weight[0:3])  # (m,n)*(n,1)
    score1 = 100 * s1 / max(s1)
    df3["Score1"] = score1

    s2 = np.dot(Data[:, 3:9], weight[3:9])  # (m,n)*(n,1)
    score2 = 100 * s2 / max(s2)
    df3["Score2"] = score2

    s3 = np.dot(Data[:, 9:12], weight[9:12])  # (m,n)*(n,1)
    score3 = 100 * s3 / max(s3)
    df3["Score3"] = score3

    s4 = np.dot(Data[:, 12:17], weight[12:17])  # (m,n)*(n,1)
    score4 = 100 * s4 / max(s4)
    df3["Score4"] = score4

    s5 = np.dot(Data[:, 17:27], weight[17:27])  # (m,n)*(n,1)
    score5 = 100 * s5 / max(s5)
    df3["Score5"] = score5

    s6 = np.dot(Data[:, 27:40], weight[27:40])  # (m,n)*(n,1)
    score6 = 100 * s6 / max(s6)
    df3["Score6"] = score6

    s7 = np.dot(Data[:, 40:44], weight[40:44])  # (m,n)*(n,1)
    score7 = 100 * s7 / max(s7)
    df3["Score7"] = score7

    s8 = np.dot(Data[:, 44:46], weight[44:46])  # (m,n)*(n,1)
    score8 = 100 * s8 / max(s8)
    df3["Score8"] = score8
    df3 = df3.fillna(100)

    df_score = date
    df_score[["Score1", "Score2", "Score3", "Score4", "Score5", "Score6", "Score7", "Score8"]] = df3[["Score1", "Score2", "Score3", "Score4", "Score5", "Score6", "Score7", "Score8"]]

    return df_score


