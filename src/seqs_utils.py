import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def check_adf(seqs: np.ndarray) -> bool:
    """
    ADF検定を行い、定常性を持つか否かを返す.

    Args:
        seqs (np.ndarray): 時系列のデータ

    Returns:
        bool: 時系列データが定常性を持つか否か
    """
    
    ctt = adfuller(seqs, regression='ctt')  # トレンド(2次)、定数あり
    ct = adfuller(seqs, regression='ct')    # トレンド(1次)、定数あり
    c = adfuller(seqs, regression='c')      # トレンドなし、定数あり
    nc = adfuller(seqs, regression='n')     # トレンド、定数なし
    # print(f'ctt p-value:\n{ctt[1]}')
    # print(f'ct p-value:\n{ct[1]}')
    # print(f'c p-value:\n{c[1]}')
    # print(f'nc p-value:\n{nc[1]}')
    
    result = adfuller(seqs)
    print(f'ADF Statistics: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical values:')
    for i, j in result[4].items():
        print('\t', i, j)
    
    return ctt[1] <= 0.05 and ct[1] <= 0.05 and c[1] <= 0.05 and nc[1] <= 0.05

def style_set() -> None:
    """ グラフのスタイル、サイズ、フォントサイズを指定 """
    sns.set()
    japanize_matplotlib.japanize()
    plt.rcParams['figure.figsize'] = [12, 9]
    plt.rcParams['font.size'] = 14

def plot_correlogram(seqs: np.ndarray, lags=12) -> None:
    """
    コレログラム(自己相関と偏自己相関)をプロットする.

    Args:
        seqs (np.ndarray): 時系列データ
        lags (int, optional): 周期数. Defaults to 12.
    """
    acf = plot_acf(seqs, lags=lags)    # 自己相関
    pacf = plot_pacf(seqs, lags=lags)  # 偏自己相関
