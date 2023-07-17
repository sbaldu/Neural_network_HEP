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

# Standardize all the feature in the dataset
for col in data:
    if col[0:2] in ['DER', 'PRI']:
        scaler = StandardScaler()
        data[col] = scaler.fit_transform(data[col])

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

data.to_csv("../data/training_processed.csv", index=None)
reduced_data.to_csv("../data/training_processed_reduced.csv", index=None)
