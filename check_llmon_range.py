import sqlite3
import pandas as pd

db_path = '/workspace/lob_databases_calibrated_volume/AMZN_2012-06-21_LLMON_calibrated.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql('SELECT MIN(mid_price) as min_p, MAX(mid_price) as max_p FROM orderbook', conn)
conn.close()

print(f'LLMON price range: ${df["min_p"].iloc[0]:.2f} - ${df["max_p"].iloc[0]:.2f}')