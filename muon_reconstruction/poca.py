import numpy as np


# Basically find the point of closest approach of the incoming and outgoing rays
# and denote that as were the muon scattered.
# [x, y, py / px, pz / px]
def poca(inputs, outputs, size=(64, 64, 64)):
    voxels = np.zeros(size)
    for i in range(len(inputs)):
        # Find the point of closest approach
        x, y, z = 0, 0, 0

        # Add scattering density
        voxels[x, y, z] = 0

    return voxels
