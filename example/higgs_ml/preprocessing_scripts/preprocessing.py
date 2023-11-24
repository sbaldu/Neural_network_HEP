import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from psynlig import pca_residual_variance
import pandas as pd

# Load the full dataset
full_dataset = pd.read_csv('../data/higgs_ml_full_data.csv')

# First of all, we want to convert the labels into integers
# We indicate signal events with 1 and background events with 0
full_dataset['Label'].replace(inplace=True, to_replace='s', value=1)
full_dataset['Label'].replace(inplace=True, to_replace='b', value=0)

# Remove the 'Kaggle' features, which are of no use
full_dataset.drop(labels=['KaggleSet', 'KaggleWeight'], axis=1, inplace=True)
# Also remove the Event_id and weight columns
full_dataset.drop(labels=['EventId', 'Weight'], axis=1, inplace=True)

# Next, we shift all the columns one position to the left, so that the labels are
# in the first column
cols = full_dataset.columns.tolist()
cols = cols[-1:] + cols[:-1]
full_dataset = full_dataset[cols]

# We want to remove the events where one or more features have the value '-999', because it means that
# the feature couldn't be computed
to_be_discarded = []
for i, row in full_dataset.iterrows():
    if -999.0 in row.values.tolist():
        to_be_discarded.append(i)
full_dataset = full_dataset.drop(to_be_discarded)
print(full_dataset)

full_dataset_len = len(full_dataset.index)
training_dataset = full_dataset.iloc[:int(0.8 * full_dataset_len), :]
validation_dataset = full_dataset.iloc[int(0.8 * full_dataset_len) : int(0.9 * full_dataset_len), :]
test_dataset = full_dataset.iloc[int(0.9 * full_dataset_len) : , :]

# Standardize all the feature in the dataset
for col in training_dataset:
    if col[0:3] in ['DER', 'PRI']:
        scaler = StandardScaler()
        training_dataset[col] = scaler.fit_transform(training_dataset[col].to_numpy().reshape(-1, 1))

        # We also normalize the validation and test datasets, with the same mean and
        # standard variance of the training dataset
        validation_dataset[col] = scaler.transform(validation_dataset[col].to_numpy().reshape(-1, 1))
        test_dataset[col] = scaler.transform(test_dataset[col].to_numpy().reshape(-1, 1))
print(training_dataset)

# Perform principal component analysis on the training dataset and plot the residual
# variance in function of the number of components
fig, ax = plt.subplots(1, 2, sharey=None)
pca_r = PCA()
pca_r.fit_transform(training_dataset.iloc[:,1:])


pca_residual_variance(pca_r, marker='o', axi=ax[0])
ax[0].grid(linewidth=0.2)

# After doing the pca, we try to plot the two categories in function of the first two
# principal components
pca = PCA(n_components=2)
training_reduced = pca_r.fit_transform(training_dataset.iloc[:,1:])
scatter = ax[1].scatter(training_reduced[:,0], training_reduced[:,1], c=training_dataset['Label'])
ax[1].grid(linewidth=0.2)
ax[1].legend(handles=scatter.legend_elements()[0], labels=['Signal', 'Background'])
ax[1].set_xlabel("First principal component")
ax[1].set_ylabel("Second principal component")
figure = plt.gcf()
figure.set_size_inches(11, 5)
plt.savefig('../tex/img/pca.png', dpi = 100)

training_dataset.to_csv('training_data.csv', index=None)
validation_dataset.to_csv('validation_data.csv', index=None)
test_dataset.to_csv('test_data.csv', index=None)
