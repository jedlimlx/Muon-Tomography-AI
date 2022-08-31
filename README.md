# Computed Tomography AI

The purpose of this AI is to perform computed tomography using x-rays / muons / other particles. 
Data is synthetic and obtained from Geant4.

## Experimental Logs

Experiments conducted are logged below.

### Muons Dataset

| Activation Function | Loss Function | Blocks          | Dropout Rate | Accuracy |
|---------------------|---------------|-----------------|--------------|----------|
| Swish               | BCE           | [1, 2, 2, 2, 3] | 0.05         | 93.9%    |
| Swish               | DL            | [1, 2, 2, 2, 3] | 0.05         | 91.9%    |
| Swish               | DL + BCE      | [1, 2, 2, 2, 3] | 0.05         |          |

DL - Dice Loss, BCE - Binary Crossentropy
