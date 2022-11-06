""" analysis python program for ROOT file ouput from geant4-example_muon """

import sys
import numpy as np
import matplotlib.pylab as plt
from ROOT import TFile
from root_numpy import root2array

def main():
    print("  All libraries were imported.")

    filename = 'out.root' # loaded file name
    fil = TFile.Open(filename,'READ')
    print(fil)
    print("  %s file was loaded."%filename)

    data = root2array(filename) # covert root to ndarray
    # data format: nhit,x,y,z,time,eIn,eDep,trackID,copyNo,particle
    print("  The total effective events (NHit>0) is : %d"%len(data))

    total_nhit = np.zeros(len(data))
    total_x = []
    total_y = []
    total_z = []
    total_edep = []
    total_id = []

    
    for i,line in enumerate(data):
        #print(line)
        total_nhit[i] = line[0]
        total_x.extend(line[1])
        total_y.extend(line[2])
        total_z.extend(line[3])
        total_edep.extend(line[6])
        total_id.extend(line[8])
        #if i==100000: break # for test

    plt.figure()
    plt.title("Number of Hits")
    x_nhit = np.arange(1,max(total_nhit))
    y_nhit = np.histogram(total_nhit,bins=x_nhit)[0]
    plt.step(x_nhit[:-1],y_nhit)
    
    plt.figure()
    plt.title("x [mm]")
    plt.hist(total_x,bins=100)

    plt.figure()
    plt.title("y [mm]")
    plt.hist(total_y,bins=100)
    
    plt.figure()
    plt.title("z [mm]")
    plt.hist(total_z,bins=100)

    plt.figure()
    plt.title("Energy deposition [MeV]")
    plt.hist(total_edep,bins=100)

    plt.figure()
    plt.title("copy No.")
    x_id = np.arange(0,128)
    y_id = np.histogram(total_id,bins=x_id)[0]
    plt.step(x_id[:-1],y_id)

    plt.show()

if __name__=='__main__':
    main()
    sys.exit(0)
