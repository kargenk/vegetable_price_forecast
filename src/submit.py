from datetime import date, datetime
from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pmdarima import auto_arima
from pmdarima.model_selection import train_test_split
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def style_set() -> None:
    """ グラフのスタイル、サイズ、フォントサイズを指定 """
    sns.set()
    japanize_matplotlib.japanize()
    plt.rcParams['figure.figsize'] = [12, 9]
    plt.rcParams['font.size'] = 14

def data_split(df: pd.DataFrame, date: date = datetime(2018, 11, 1),
               plot: bool = False) -> list[pd.DataFrame]:
    """
    学習データとテストデータ(直近12ヵ月)に分割

    Args:
        date (date, optional): 境目の日付け. Defaults to datetime(2018, 11, 1).
        plot (bool, optional): 訓練と検証データの境目画像を作成するか. Defaults to False.

    Returns:
        list[pd.DataFrame]: 訓練データと検証データ
    """
    df_train, df_val = train_test_split(df, test_size=12)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(df_train.index, df_train.values, label='train')
        ax.plot(df_val.index, df_val.values, label='val', color='red')
        ax.axvline(date, color='blue')
        plt.title('取引価格')
        plt.xlabel('Month')
        plt.ylabel('Monthly price of Vegetable')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR.joinpath('data_split.png'))
    
    return df_train, df_val

def evaluate(pred: pd.Series, y: pd.Series) -> None:
    """
    RMSE(二乗平均平方根誤差)、MAE(平均絶対誤差)、MAPE(平均絶対パーセント誤差)を算出する.

    Args:
        pred (pd.Series): モデルの予測値
        y (pd.Series): 真の値
    """

if __name__ == '__main__':
    ROOT_DIR = Path(__file__).parents[1]
    DATA_DIR = ROOT_DIR.joinpath('data')
    OUTPUT_DIR = ROOT_DIR.joinpath('outputs')
    model_name = 'holtwinters'
    
    style_set()
    
    # 提出用ファイル
    df_submit = pd.read_csv(DATA_DIR.joinpath('submission.csv'))
    
    # 対数変換して軸名も変更
    df = pd.read_csv(DATA_DIR.joinpath('train_data.csv'), index_col='id', parse_dates=True)
    df = np.log1p(df).rename_axis(index='date', columns='vegetable')

    preds = []
    for col in df.columns:
        print(col)
        match model_name:
            case 'arima':
                # 季節性を考慮し、周期を1年: 12とするモデルを構築
                # SARIMAX(2, 0, 0)(1, 1, 0)[12]: 非季節性パラメータのARの次数が2、季節性パラメータのARの次数が1、SIの次数が1
                model = auto_arima(df[col], seasonal=True, m=12)
                val_pred = model.predict(n_periods=1)
            case 'holtwinters':
                # 加法モデルを仮定
                model = ExponentialSmoothing(df[col],
                                            trend='additive',
                                            seasonal='additive',
                                            seasonal_periods=12)
                model = model.fit()
                val_pred = model.forecast(1)  # hw
            case 'prophet':
                # データ作成
                _df = pd.DataFrame({'ds': df.index,
                                    'y': df[col]})
                # モデルの学習
                model = Prophet()
                model.fit(_df)
                # 予測
                df_future = model.make_future_dataframe(periods=1, freq='M')
                df_pred = model.predict(df_future)
                val_pred = df_pred['yhat'].iloc[-1:]

        # 予測値をもとのスケールに変換
        pred = np.expm1(val_pred.values[0])
        print(pred)
        preds.append(pred)
    
    df_submit['y'] = preds
    df_submit.to_csv(OUTPUT_DIR.joinpath(f'sub_{model_name}.csv'),
                     index=False)
