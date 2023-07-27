import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from psynlig import pca_residual_variance
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

# Perform principal component analysis and plot the residual variance in function
# of the number of components
pca_r = PCA()
pca_r.fit_transform(data)

pca_residual_variance(pca_r, marker='o')
plt.grid(linewidth=0.2)
figure = plt.gcf()
figure.set_size_inches(8, 5)
plt.savefig('../tex/img/residual_variance.png', dpi = 100)

# components = []
# variance_ratios = []
# for i in range(1, 31):
#     pca_i = PCA(n_components=i)
#     pca_i.fit_transform(data)
#     components.append(i)
#     variance_ratios.append(np.sum(pca_i.explained_variance_ratio_))
# plt.plot(components, variance_ratios, marker='o')
# plt.xlabel('Number of components')
# plt.ylabel('Explained variance ratio')
# plt.grid(linewidth=0.2)
# plt.show()

# data.to_csv("../data/training_filtered_processed.csv", index=None)
# reduced_data.to_csv("../data/training_filtered_processed_reduced.csv", index=None)
# data.to_csv("../data/training_processed.csv", index=None)
# reduced_data.to_csv("../data/training_processed_reduced.csv", index=None)
