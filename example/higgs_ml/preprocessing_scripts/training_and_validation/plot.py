import matplotlib.pyplot as plt
from glob import glob

fig1, ax1 = plt.subplots(2,2, sharey=None)

files = glob("../data/accuracy_data/values_*_1.dat")
# Sort files by the number of neurons
files.sort()
files =  files[1:] + files[0:1]

for file in files:
    y_values = []
    f = open(file, 'r')
    for line in f:
        y_values.append(float(line))
    ax1[0,0].plot(range(1, len(y_values)+1), y_values)

legend = []
for file in files:
    split_name = file.split('/')
    num = split_name[3].split('.')[0].split('_')[1]
    legend.append(str(num) + ' neurons')

ax1[0,0].legend(legend)
ax1[0,0].grid(linewidth=0.2)
ax1[0,0].set_xlabel("Number of epochs")
ax1[0,0].set_ylabel("Accuracy (%)")
ax1[0,0].set_title("$\eta$ = 0.97")

files = glob("../data/accuracy_data/values_*_2.dat")
# Sort files by the number of neurons
files.sort()
files =  files[1:] + files[0:1]

for file in files:
    y_values = []
    f = open(file, 'r')
    for line in f:
        y_values.append(float(line))
    ax1[0,1].plot(range(1, len(y_values)+1), y_values)

legend = []
for file in files:
    split_name = file.split('/')
    num = split_name[3].split('.')[0].split('_')[1]
    legend.append(str(num) + ' neurons')

ax1[0,1].legend(legend)
ax1[0,1].grid(linewidth=0.2)
ax1[0,1].set_xlabel("Number of epochs")
ax1[0,1].set_ylabel("Accuracy (%)")
ax1[0,1].set_title("$\eta$ = 0.975")

files = glob("../data/accuracy_data/values_*_3.dat")
# Sort files by the number of neurons
files.sort()
files =  files[1:] + files[0:1]

for file in files:
    y_values = []
    f = open(file, 'r')
    for line in f:
        y_values.append(float(line))
    ax1[1,0].plot(range(1, len(y_values)+1), y_values)

legend = []
for file in files:
    split_name = file.split('/')
    num = split_name[3].split('.')[0].split('_')[1]
    legend.append(str(num) + ' neurons')

ax1[1,0].legend(legend)
ax1[1,0].grid(linewidth=0.2)
ax1[1,0].set_xlabel("Number of epochs")
ax1[1,0].set_ylabel("Accuracy (%)")
ax1[1,0].set_title("$\eta$ = 0.98")

files = glob("../data/accuracy_data/values_*_4.dat")
# Sort files by the number of neurons
files.sort()
files =  files[1:] + files[0:1]

for file in files:
    y_values = []
    f = open(file, 'r')
    for line in f:
        y_values.append(float(line))
    ax1[1,1].plot(range(1, len(y_values)+1), y_values)

legend = []
for file in files:
    split_name = file.split('/')
    num = split_name[3].split('.')[0].split('_')[1]
    legend.append(str(num) + ' neurons')

ax1[1,1].legend(legend)
ax1[1,1].grid(linewidth=0.2)
ax1[1,1].set_xlabel("Number of epochs")
ax1[1,1].set_ylabel("Accuracy (%)")
ax1[1,1].set_title("$\eta$ = 0.985")
figure = plt.gcf()
figure.set_size_inches(14, 10)
plt.savefig('../tex/img/accuracy_plots.png', dpi = 100)
# plt.savefig('./fig.png')

fig2, ax2 = plt.subplots(2,2, sharey=None)

training_loss_files = glob("../data/training_loss_data/training_loss_*_1.dat")
# Sort files by the number of neurons
training_loss_files.sort()
training_loss_files = training_loss_files[1:] + training_loss_files[0:1]

for file in training_loss_files:
    y_values = []
    f = open(file, 'r')
    for line in f:
        y_values.append(float(line))
    ax2[0,0].plot(range(1, len(y_values)+1), y_values)

legend = []
for file in training_loss_files:
    split_name = file.split('/')
    num = split_name[3].split('.')[0]
    num = split_name[3].split('.')[0].split('_')[2]
    legend.append(str(num) + ' neurons')

ax2[0,0].legend(legend)
ax2[0,0].grid(linewidth=0.2)
ax2[0,0].set_xlabel("Number of epochs")
ax2[0,0].set_ylabel("Training loss")
ax2[0,0].set_title("$\eta$ = 0.97")

training_loss_files = glob("../data/training_loss_data/training_loss_*_2.dat")
# Sort files by the number of neurons
training_loss_files.sort()
training_loss_files = training_loss_files[1:] + training_loss_files[0:1]

for file in training_loss_files:
    y_values = []
    f = open(file, 'r')
    for line in f:
        y_values.append(float(line))
    ax2[0,1].plot(range(1, len(y_values)+1), y_values)

legend = []
for file in training_loss_files:
    split_name = file.split('/')
    num = split_name[3].split('.')[0]
    num = split_name[3].split('.')[0].split('_')[2]
    legend.append(str(num) + ' neurons')

ax2[0,1].legend(legend)
ax2[0,1].grid(linewidth=0.2)
# ax2[0,1].set_xticks(np.arange(0, 36, 2.5))
ax2[0,1].set_xlabel("Number of epochs")
ax2[0,1].set_ylabel("Training loss")
ax2[0,1].set_title("$\eta$ = 0.975")

training_loss_files = glob("../data/training_loss_data/training_loss_*_3.dat")
# Sort files by the number of neurons
training_loss_files.sort()
training_loss_files = training_loss_files[1:] + training_loss_files[0:1]

for file in training_loss_files:
    y_values = []
    f = open(file, 'r')
    for line in f:
        y_values.append(float(line))
    ax2[1,0].plot(range(1, len(y_values)+1), y_values)

legend = []
for file in training_loss_files:
    split_name = file.split('/')
    num = split_name[3].split('.')[0]
    num = split_name[3].split('.')[0].split('_')[2]
    legend.append(str(num) + ' neurons')

ax2[1,0].legend(legend)
ax2[1,0].grid(linewidth=0.2)
# ax2[1,0].set_xticks(np.arange(0, 36, 2.5))
ax2[1,0].set_xlabel("Number of epochs")
ax2[1,0].set_ylabel("Training loss")
ax2[1,0].set_title("$\eta$ = 0.98")

training_loss_files = glob("../data/training_loss_data/training_loss_*_4.dat")
# Sort files by the number of neurons
training_loss_files.sort()
training_loss_files = training_loss_files[1:] + training_loss_files[0:1]

for file in training_loss_files:
    y_values = []
    f = open(file, 'r')
    for line in f:
        y_values.append(float(line))
    ax2[1,1].plot(range(1, len(y_values)+1), y_values)

legend = []
for file in training_loss_files:
    split_name = file.split('/')
    num = split_name[3].split('.')[0].split('_')[2]
    legend.append(str(num) + ' neurons')

ax2[1,1].legend(legend)
ax2[1,1].grid(linewidth=0.2)
# ax2[1,1].set_xticks(np.arange(0, 36, 2.5))
ax2[1,1].set_xlabel("Number of epochs")
ax2[1,1].set_ylabel("Training loss")
ax2[1,1].set_title("$\eta$ = 0.985")
figure = plt.gcf()
figure.set_size_inches(14, 10)
plt.savefig('../tex/img/training_loss_plots.png', dpi = 100)
