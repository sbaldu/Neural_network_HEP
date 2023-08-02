import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1,2, sharey=None)

accuracy_file = open('./mnist_accuracy.dat', 'r')
accuracy = []
for line in accuracy_file:
    accuracy.append(float(line))
ax[0].plot(np.arange(1, 16), accuracy)
ax[0].grid(linewidth=0.2)
ax[0].set_xlabel("Number of epochs")
ax[0].set_ylabel("Accuracy (%)")

loss_file = open('./mnist_loss.dat', 'r')
loss = []
for line in loss_file:
    loss.append(float(line))
ax[1].plot(np.arange(1, 16), loss)
ax[1].grid(linewidth=0.2)
ax[1].set_xlabel("Number of epochs")
ax[1].set_ylabel("Loss")

# Save plots in png file
figure = plt.gcf()
figure.set_size_inches(10, 4)
plt.savefig('../../../../tex/img/mnist.png', dpi = 100)
