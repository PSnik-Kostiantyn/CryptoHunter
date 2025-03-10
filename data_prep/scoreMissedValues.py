import pandas as pd

file_path = "/home/kostiantyn/PycharmProjects/CryptoHunter/analyzed_news_res_2.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

start_date = "2023-01-01 02:00:00"
end_date = "2025-01-01 00:00:00"
date_range = pd.date_range(start=start_date, end=end_date, freq='h')

full_df = pd.DataFrame({'date': date_range})
full_df['timestamp'] = full_df['date'].astype(int) // 10**9

merged_df = full_df.merge(df, on=['timestamp', 'date'], how='left')

missing_dates = merged_df[merged_df['score'].isna()]['date']
print("Пропущені дати:")
print(missing_dates)

