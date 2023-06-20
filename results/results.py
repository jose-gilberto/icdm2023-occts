# %%
import pandas as pd

pd.read_csv('./ucr_dagmm.csv').accuracy.mean()
# %%
pd.read_csv('./ucr_deepsvdd.csv').accuracy.mean()
# %%
pd.read_csv('./ucr_resoc.csv').accuracy.mean()

# %%
