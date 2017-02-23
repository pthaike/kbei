import pandas as pd
import numpy as np
import pdb

def pay():
    dic = {}
    fr = open('../user_pay.txt','r')
    start = pd.datetime(2015,7,1)
    for line in fr.readlines():
        line = line.strip()
        li = line.split(',')
        key = li[1].strip()
        if(not dic.has_key(key)):
            print key
            dic[key] = np.zeros(489)
        d = pd.to_datetime(li[2].strip())
        gap = (d - start).days
        dic[key][gap] += 1
    fr.close()
    fw = open('../pay1.csv','w')
    print 'read finish'
    for k in dic.keys():
        fw.write(k)
        fw.write(',')
        flag = False
        for n in dic[k]:
            if (flag):
                fw.write(str(int(n)))
                fw.write(',')
                continue
            if n != 0:
                flag = True
        fw.write('\n')
    fw.close()

def pay1():
    fr = open("../pay1.csv",'r')
    fw = open("../pay_no0.csv", 'w')
    for line in fr.readlines():
        line = line.strip()
        ls = line.split(",")
        fw.write(ls[0]+",")
        for l in ls[1:]:
            if l == '0':
                continue
            fw.write(l+",")
        fw.write('\n')
    fw.close()
    fr.close()


def num():
    fr = open("../pay2.csv",'r')
    m = 100000
    for line in fr.readlines():
        line = line.strip()
        ls = line.split(",")
        m = min(len(ls), m)
        print len(ls)
    fr.close()
    print m

def lastavg(n):
    fr = open("../pay_no0.csv",'r')
    fw = open("../pay_last3.csv", 'w')
    for line in fr.readlines():
        line = line.strip()
        ls = line.split(",")
        m = len(ls)
        fw.write(ls[0]+',')
        for i in range(m-n, m):
            if ls[i] == "":
                continue
            l = (int(ls[i]) + int(ls[i-1]) + int(ls[i-2])) / 3
            if l != "":
                fw.write(str(int(l)) + ',')
        fw.write('\n')
    fw.close()
    fr.close()

def getdata(file):
    df = pd.read_csv(file, header = None)
    y = df.values[:,0]
    x = df.values[:,1:]
    return x,y

"""
get training data
data:data
l: length that return of feature x
"""
def gettrain(data, l):
    m,n = np.shape(data)
    k = m*(n-l)
    x = np.zeros((k, l))
    y = np.zeros(k)
    ix = 0
    for i in range(m):
        for j in range(n-l):
            x[ix] = data[i][j:j+l]
            y[ix] = data[i][j+l]
            ix += 1
    return x,y

"""
combine x and y
"""
def getnexttest(x, prex):
    m,n = np.shape(x)
    for i in range(m):
        for j in range(n-1):
            x[i][j] = x[i][j+1]
        x[i][j+1] = prex[i]
    return x



def gettestx(data, l):
    m,n = np.shape(data)
    if l > n:
        print "number of feature greater than data"
        return
    x = np.zeros((m,l))
    for i in range(m):
        x[i] = data[i,n-l:]
    return x



if __name__ == '__main__':
    # x,did = getdata('x.txt')
    # print gettestx(x,2)
    # x,y = gettrain(x,2)
    # print x
    # print y

    # print getnexttrain(x,y)

    # pay1();
    lastavg(80)
