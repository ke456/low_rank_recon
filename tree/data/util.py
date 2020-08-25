def print_data(name, data):
    f = open(name, 'w')
    for d in data:
        for c in range(len(d)):
            f.write(str(d[c]))
            if (c == len(d)-1):
                f.write('\n')
            else:
                f.write(',')
    f.close()
    
def normalize(data):
    for f in range(len(data[0])):
        min_val = data[0][f]
        max_val = data[0][f]
        for d in data:
            if d[f] > max_val:
                max_val = d[f]
            if d[f] < min_val:
                min_val = d[f]
        for d in data:
            d[f] = (d[f] - min_val)/(max_val-min_val)