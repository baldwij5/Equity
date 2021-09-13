import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import pandas as pd

ei_data = pd.read_csv(r'data/IL_Cook_MedIncWeight.csv')

# input
ind_cols = [
  'Normalized Educational Attainment',
  'Normalized Non-White',
  'Normalized Non-English Speaking',
  'Normalized Poverty Score',
  'Normalized Median Income',
]

dep_col = 'Equity Index'


# clean data
# [print(ei_data[x].value_counts(dropna=False).reset_index()) for x in ind_cols] # look for null values
ei_data[dep_col].value_counts(dropna=False).reset_index()
ei_data.loc[ei_data[dep_col].isna(), ind_cols]

ei_data.dropna(subset=ind_cols, inplace=True)

print("Number of rows: ", len(ei_data))



# convert to numpy
x = ei_data[ind_cols].to_numpy()
x = sm.add_constant(x) # add constant for stats models to identify intercept
y = ei_data[dep_col].to_numpy()


# split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# train
model = sm.OLS(y_train, x_train)

# initial model results
results = model.fit()
print(results.summary())
y_train_pred = results.predict(x_train)


#test
y_pred = results.predict(x_test)

test_compare_df = pd.DataFrame({
  "actual" : y_test,
  "predicted" : y_pred,
})

test_compare_df['difference'] = test_compare_df['actual'] - test_compare_df['predicted']

test_compare_df['difference'].value_counts(dropna=False).reset_index()

import matplotlib.pyplot as plt

num_bins = 8
n, bins, patches = plt.hist(test_compare_df['difference'], num_bins, facecolor='blue', alpha=0.5)
plt.show()

# almost no difference in result
# model foudn the formula for the equity index, no need to use prediction

print("Variables key for x1 to xn")
[print(i) for i in ei_data[ind_cols]]