import datetime
import pandas as pd
import numpy as np
# import boto3
# import sagemaker
import feature_convert
# from snowflake import SnowflakeQueryExecutor

data = pd.read_csv('/Users/shuwen/Downloads/justfortest.csv')
df = feature_convert.percentile_encoding(data, 'RENDER_NUMBER')
df = feature_convert.one_hot_encoding(df, ['AD_TYPE', 'DAY'])
df = feature_convert.frequency_encoding(df, 'COUNTRY_GROUP')
df = feature_convert.label_encoding(df, )