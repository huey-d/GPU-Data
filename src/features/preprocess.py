import pandas as pd
import sqlite3
from prep_path import db_path

query = """
SELECT
  *,
  CASE
    WHEN productname LIKE '%RTX%'
    OR productname LIKE '%GeForce%' THEN 'NVIDIA'
    WHEN productname LIKE '%RX%'
    OR productname LIKE '%Radeon%' THEN 'AMD'
  END AS chip_make
FROM
  gpu_tb
WHERE
  url NOT IN (
    SELECT
      url
    FROM
      gpu_tb
    WHERE
      url LIKE 'https://www.newegg.com/Product/%'
  )
ORDER BY
  rank
    """

def preprocess_data():
    Brand = ['ASUS', 'MSI', 'EVGA', 'GIGABYTE', 'NVIDIA', 'SAPPHIRE', 'ASROCK', 'PNY', 'ZOTAC', 'XFX', 'VISIONTEK', 'DELL', 'HP', 'MAXSUN', 'POWERCOLOR', 'YESTON', 'VISIONTEK']


    con = sqlite3.connect(db_path)
    df = pd.read_sql(query, con)
    
    df = df.set_index('rank')

    productname = df['productname'].str.split(expand=True)
    rating = df['rating'].str.split(expand=True)

    df['brand'] = productname[0].str.upper()
    df['Rating'] = rating[2]

    df['chip_make'] = df['chip_make'].fillna(df['chipmake'])
    df = df.drop(['chipmake', 'rating'], axis=1)
    df = df[df['brand'].isin(Brand)]
    
    df = df.dropna(subset=['price', 'chip_make'])
    df['Rating'] = df['Rating'].fillna(value='No Rating')
    df.to_csv('data/processed/gpu_data.csv')

if __name__ == '__main__':
    preprocess_data()