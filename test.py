from google.colab import files
uploaded = files.upload()

import pandas as pd
credit_data = pd.read_excel('Credit data.xlsx')
import numpy as np
# 通过zScore分数删除异常行
def use_zscore_find_outliner(data, varabiles):
    current_colume = data[varabiles]
    std = int(np.std(current_colume, ddof=1))
    index = -1
    arr = []
    print(np.std(current_colume, ddof=1), std)
    if (std == 0):
        return

    for col in current_colume:
        index = index + 1
        a = (col - current_colume.mean()) / np.std(current_colume, ddof=1)
        if ((abs(a) > 3)):
            print(abs(a), index)
            arr.append(index)
    if (arr):
        global credit_data
        credit_data = credit_data.drop(index=arr)


from sklearn.ensemble import RandomForestClassifier


def use_random_forest(df1):
    # 把已有的数值型特征取出来
    df = pd.DataFrame(df1)
    print()
    process_df = df.ix[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df.MonthlyIncome.notnull()].as_matrix()
    unknown = process_df[process_df.MonthlyIncome.isnull()].as_matrix()
    # X为特征属性值
    X = known[:, 1:]
    # y为结果标签值
    y = known[:, 0]
    # fit到RandomForestRegressor之中
    rfr = RandomForestClassifier(random_state=0,
                                 n_estimators=200, max_depth=3, n_jobs=-1)
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    print(predicted)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df


from sklearn.ensemble import RandomForestRegressor


def fill_missing_rf(data, name):
    df1 = data[data[name].isnull()]
    df2 = data[~data[name].isnull()]
    X1 = df1.drop(columns=name)
    X2 = df2.drop(columns=name)
    y = df2[name]

    RF = RandomForestRegressor(max_depth=10, ).fit(X2, y)
    return RF.predict(X1)  # 返回预测值
    # 对缺少部分进行重新赋值
    data.loc[data['MonthlyIncome'].isnull(), 'MonthlyIncome'] = fill_missing_rf(data, 'MonthlyIncome')
    # 检查是否还有缺少情况
    data.isnull().any()
