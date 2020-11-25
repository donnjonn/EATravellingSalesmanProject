import csv
import matplotlib.pyplot as plt
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=100_alpha=0.1_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='alpha = 0.1')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=100_alpha=0.01_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='alpha = 0.01')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=100_alpha=0.5_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='alpha = 0.5')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=100_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='alpha = 0.05')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()



plt.figure()
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=50_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='lambda=50')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=100_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='lambda=100')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=200_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='lambda=200')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=300_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='lambda=300')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=400_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='lambda=400')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=500_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='lambda=500')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=600_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='lambda=600')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=1000_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='lambda=1000')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()




plt.figure()
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=500_alpha=0.025_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='alpha=0.025')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=500_alpha=0.05_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='alpha=0.05')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=500_alpha=0.075_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='alpha=0.075')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=500_alpha=0.1_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='alpha=0.1')
x= []
y= []
with open ('r0123456iter=3000_stopcrit200_lambda=500_alpha=0.1_k=5_tour29.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in plots:
        if count < 2:
            count += 1
            continue
        count += 1
        print(row[0])
        x.append(float(row[0]))
        y.append(float(row[2]))
plt.plot(x,y, label='alpha=0.2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()