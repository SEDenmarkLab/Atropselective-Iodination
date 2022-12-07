import pandas as pd
from pandas.io.formats.style import jinja2

df = pd.read_csv("in_silico_pred_split0.csv", index_col=[0])
print(df)
dfg47 = df.loc[[label for label in df.index if label.split('_')[-1] == "G47"], : ]
print(dfg47)

df_sorted_delta_g = dfg47.sort_values(by="y_predict", ascending=False)
print(df_sorted_delta_g)

df_sorted_delta_g_first_100 = df_sorted_delta_g.head(100)
print(df_sorted_delta_g_first_100)

df_sorted_delta_g_first_100.to_csv('ranked_top_50_cats_split0.csv')