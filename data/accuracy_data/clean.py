from glob import glob

files = glob('./values*.txt')

for file in files:
    f = open(file, 'r')
    new_name = file.split('.')[1][1:] + '.dat'
    new_file = open(new_name, 'w')
    for line in f:
        if '-' in line:
            continue 
        new_file.write(line.split(' ')[2] + '\n')

f.close()
new_file.close()
