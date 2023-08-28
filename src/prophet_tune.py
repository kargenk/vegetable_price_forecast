from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from pmdarima.model_selection import train_test_split
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit


def data_split(df: pd.DataFrame, test_size: int = 11,
               date: date = datetime(2018, 11, 1),
               plot: bool = False) -> list[pd.DataFrame]:
    """
    学習データとテストデータ(直近{test_size}ヵ月)に分割

    Args:
        test_size (int, optional): テストサイズ. Defaults to 11.
        date (date, optional): 境目の日付け. Defaults to datetime(2018, 11, 1).
        plot (bool, optional): 訓練と検証データの境目画像を作成するか. Defaults to False.

    Returns:
        list[pd.DataFrame]: 訓練データと検証データ
    """
    df_train, df_val = train_test_split(df, test_size=test_size)
    
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
    pass

def objective(trial):
    # ハイパーパラメータの探索範囲
    params = {
        'changepoint_prior_scale':
            trial.suggest_float('changepoint_prior_scale', 0.001, 0.5),
        'seasonality_prior_scale':
            trial.suggest_float('seasonality_prior_scale', 0.01, 10),
        'seasonality_mode':
            trial.suggest_categorical('seasonality_mode',
                                      ['additive', 'multiplicative']),
        'changepoint_range':
            trial.suggest_float('changepoint_range', 0.8, 0.95, step=0.001),
        'n_changepoints':
            trial.suggest_int('n_changepoints', 20, 35),
    }
    
    # 時系列CV
    tcv = TimeSeriesSplit(test_size=1)
    cv_mse = []
    
    for fold, (train_index, valid_index) in enumerate(tcv.split(df_train)):
        # データの分割
        train_data = df_train.iloc[train_index]
        valid_data = df_train.iloc[valid_index]
        
        # モデルの学習
        model = Prophet(**params)
        model.fit(train_data)
        
        # 予測
        df_future = model.make_future_dataframe(periods=len(valid_data), freq='M')
        df_pred = model.predict(df_future)
        preds = df_pred.tail(len(valid_data))
        
        # 精度評価(MSE)
        val_mse = mean_squared_error(valid_data.y. preds.yhat)
        cv_mse.append(val_mse)
    
    return np.mean(cv_mse)

if __name__ == '__main__':
    ROOT_DIR = Path(__file__).parents[1]
    DATA_DIR = ROOT_DIR.joinpath('data')
    OUTPUT_DIR = ROOT_DIR.joinpath('outputs')
    
    # style_set()
    
    # 対数変換して軸名も変更
    df = pd.read_csv(DATA_DIR.joinpath('train_data.csv'))
    _df = df[['id', 'えのきだけ_中国']].copy()
    _df['えのきだけ_中国'] = np.log1p(_df['えのきだけ_中国'])
    _df.columns = ['ds', 'y']  # Prophetの仕様上、カラム名はdsとy
    df_train, df_val = data_split(_df, test_size=1, plot=False)
    
    model = Prophet()
    model.fit(df)
    
    # 学習したモデルで予測
    df_future = model.make_future_dataframe(periods=1, freq='M')
    df_pred = model.predict(df_future)
    val_pred = df_pred['yhat'].iloc[-1:]
    print(val_pred)
