import pandas as pd

dataset_path = '../datasets/BTC_ready_res.csv'
df = pd.read_csv(dataset_path)

df.columns = df.columns.str.strip()

df['Close time'] = pd.to_datetime(df['Close time'], format='mixed')

df['Close time'] = df['Close time'].dt.ceil('H')

df['Close time'] = df['Close time'].astype('int64') // 10**9

print(df[['Close time']].head())

df.to_csv('../datasets/BTC_ready_res.csv', index=False)