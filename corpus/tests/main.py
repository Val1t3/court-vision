import numpy as np
import pandas as pd
from typing import Dict

def load_gt(gt_csv: str) -> pd.DataFrame:
    # expected: frame,id,x,y,cum_distance (meters) or at least cum_distance
    return pd.read_csv(gt_csv)

def total_distance_error(df_est: pd.DataFrame, df_gt: pd.DataFrame) -> pd.DataFrame:
    # join on frame,id
    j = pd.merge(df_est, df_gt[['frame','id','cum_distance']].rename(columns={'cum_distance':'cum_gt'}),
                 on=['frame','id'], how='inner')
    j['abs_err'] = (j['cum_distance'] - j['cum_gt']).abs()
    # per-player aggregates
    per_player = j.groupby('id').agg(
        tde_last=('abs_err','last'),
        mae=('abs_err','mean'),
        rmse=('abs_err', lambda x: np.sqrt(np.mean(x**2)))
    ).reset_index()
    overall = {
        'TDE_last_mean': per_player['tde_last'].mean(),
        'MAE_mean': per_player['mae'].mean(),
        'RMSE_mean': per_player['rmse'].mean()
    }
    return per_player, overall

def smoothness_variance(df_est: pd.DataFrame) -> pd.DataFrame:
    df = df_est.sort_values(['id','frame']).copy()
    df['dx'] = df.groupby('id')['x_est'].diff()
    df['dy'] = df.groupby('id')['y_est'].diff()
    df['disp'] = np.sqrt(df['dx']**2 + df['dy']**2)
    sm = df.groupby('id')['disp'].std().reset_index().rename(columns={'disp':'std_disp'})
    return sm

def jitter_index(df_est: pd.DataFrame) -> pd.DataFrame:
    df = df_est.sort_values(['id','frame']).copy()
    df['dx'] = df.groupby('id')['x_est'].diff()
    df['dy'] = df.groupby('id')['y_est'].diff()
    df['ddx'] = df.groupby('id')['dx'].diff()
    df['ddy'] = df.groupby('id')['dy'].diff()
    df['j'] = np.sqrt((df['ddx']**2 + df['ddy']**2)).fillna(0)
    ji = df.groupby('id')['j'].sum().reset_index().rename(columns={'j':'jitter'})
    return ji

def runtime_summary(runtimes: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{'algo':k,'seconds':v} for k,v in runtimes.items()])
