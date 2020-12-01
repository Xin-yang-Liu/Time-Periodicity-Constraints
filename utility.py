import sys
import os
import numpy as np

def readInputData(InputFile):
    """Function is to split user input parameters make it to a dict
    
    Arg: 
    InputFile: user specifed input file
    
    Return:
    paramDict: parameter dictionary
    """
    paramDict = {}
    try:
        f = open(InputFile, 'r')
    except IOError:
        print ('cannot open the file: ', InputFile)
        sys.exit(1)
    else:
        # finDA = open(InputFile, "r")
        
        for line in f:
            # ignore empty or comment lines
            if line.isspace() or line.startswith("#"):
                continue

            # allow colon ":" in input dictionary for convenience
            param, value = line.strip().replace(':',' ').split(None, 1)
            try:
                value = float(value)
            except:
                pass
            else:
                if value.is_integer():
                    value = int(value)
            paramDict[param] = value
        f.close()
    return paramDict


def chmkdir(dirs):
    for name in dirs:
        try:
            os.chdir(name)
        except:
            os.mkdir(name)
            os.chdir(name)

def read_of(path, quantity):
    '''
    return numpy
    ====
    '''
    c = np.zeros([quantity, 10])
    for i in range(1,quantity):
        a = format(i*0.6,'.1f')
        if a[-1] == '0' and a[-2]=='.':
            a = a[0:-2]
        f = open(path + '/' + a + '/T', mode='r')
        line = f.readlines()[19]
        line = line[43:-3]
        line = np.array(line.split(' ')).astype(np.float)
        c[i,:]= line
    return np.array(c).astype(np.float)

def read2D(path):
    c = np.zeros([20, 10, 50, 3])
    for i in range(1,21):
        a = format(i*0.5,'.1f')
        if a[-1] == '0' and a[-2]=='.':
            a = a[0:-2]
        f = open(path + '/' + a + '/U', mode='r')
        line = f.readlines()[21:523][1:-1]
        f.close()
        for j in range(len(line)):
            line[j] = line[j][1:-2].split(' ')
            for k in range(3):
                line[j][k] = float(line[j][k])
        line = np.array(line)
        c[i-1]= line.reshape(10,50,3)
    return c

class a():
    def __init__(self):
        self.b=1


if __name__=='__main__':
    pass
    