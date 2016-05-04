

with open('credit-card-clients.csv', 'r') as f:
    lines = []
    for line in f:
        vals = line.strip().split(',')
        if vals[-1] == "0":
            vals[-1] = -1
        lines.append(vals)
    with open('percept-credit-card-clients.csv', 'w') as o:
        for line in lines:
            for item in line:
                print(item, file=o, end=", ")
            print("", file=o)

