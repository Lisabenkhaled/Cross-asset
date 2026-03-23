# %%
import pandas as pd

# %%
df = pd.read_parquet("../data/master_data_daily.parquet")
df.tail()
# %%
df.columns
