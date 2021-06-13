import pandas as pd

df = pd.read_excel("meta/micro_metadata_5.xlsx", index_col=0, engine='openpyxl', header=0)
print(df.shape)
df.drop(df.loc[df['primary_microconstituent'] == "figure"].index, inplace=True)
print(df.shape)
df.to_excel("meta/micro_metadata_6.xlsx")

