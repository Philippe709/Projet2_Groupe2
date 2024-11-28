import pandas as pd


df1 = pd.read_csv('./BD_a_ignorer/name.basics.tsv.gz', compression = 'gzip', sep='\t', nrows = 1000)
print(df1.sample(5))

df2 = pd.read_csv('./BD_a_ignorer/title.akas.tsv.gz', compression = 'gzip', sep='\t', nrows = 1000)
print(df2.sample(5))

df3 = pd.read_csv('./BD_a_ignorer/title.episode.tsv.gz', compression = 'gzip', sep='\t', nrows = 1000)
print(df3.sample(5))