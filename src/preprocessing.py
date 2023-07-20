import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

data = pd.read_csv("../data/training.csv")
print(data.head())

# First of all, we want to convert the labels into integers
# We indicate signal events with 1 and background events with 0
data['Label'].replace(inplace=True, to_replace='s', value=1)
data['Label'].replace(inplace=True, to_replace='b', value=0)

# Next, we shift all the columns one position to the left, so that the labels are
# in the first column
cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]

# We want to remove the events where one or more features have the value '-999', because it means that
# the feature couldn't be computed
to_be_discarded = []
for i, row in data.iterrows():
    if -999.0 in row.values.tolist():
        to_be_discarded.append(i)
data = data.drop(to_be_discarded)
print(data)

# Standardize all the feature in the dataset
for col in data:
    if col[0:3] in ['DER', 'PRI']:
        scaler = StandardScaler()
        data[col] = scaler.fit_transform(data[col].to_numpy().reshape(-1, 1))

# Remove the Event_id and weight columns since it is of no use
data.drop(labels=['EventId', 'Weight'], axis=1, inplace=True)

pca = PCA(n_components=10)
label_columns = data['Label']
reduced_data = pca.fit_transform(data)
reduced_data = pd.DataFrame(reduced_data)
print(f"The total explained variance ration is {np.sum(pca.explained_variance_ratio_)}")
print(data)
print(reduced_data)
reduced_data = pd.concat([label_columns, reduced_data], axis=1)
print(reduced_data)

data.to_csv("../data/training_filtered_processed.csv", index=None)
reduced_data.to_csv("../data/training_filtered_processed_reduced.csv", index=None)
# data.to_csv("../data/training_processed.csv", index=None)
# reduced_data.to_csv("../data/training_processed_reduced.csv", index=None)
