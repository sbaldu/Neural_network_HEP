import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

accuracy = pd.read_csv('./training_validation_accuracy_975_300.csv', header=None)
loss = pd.read_csv('./training_validation_loss_975_300.csv', header=None)

fig, ax = plt.subplots(1, 2, sharey=None)

ax[0].grid(linewidth=0.2)
ax[0].plot(np.arange(1, 36), accuracy[0])
ax[0].plot(np.arange(1, 36), accuracy[1])
ax[0].legend(['Training', 'Validation'])
ax[0].set_xlabel('Number of epochs')
ax[0].set_ylabel('Accuracy (%)')

ax[1].grid(linewidth=0.2)
ax[1].plot(np.arange(1, 36), loss[0])
ax[1].plot(np.arange(1, 36), loss[1])
ax[1].legend(['Training', 'Validation'])
ax[1].set_xlabel('Number of epochs')
ax[1].set_ylabel('Loss')

figure = plt.gcf()
figure.set_size_inches(12, 5)
plt.savefig('../tex/img/training_validation.png', dpi = 100)
