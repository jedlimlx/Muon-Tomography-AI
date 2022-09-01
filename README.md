# Computed Tomography AI

The purpose of this AI is to perform computed tomography using x-rays / muons / other particles. 
Data is synthetic and obtained from Geant4.

## Experimental Logs

Experiments conducted are logged below.

### Muons Dataset

40 Epochs

| Activation Function | Loss Function        | Blocks          | Dropout Rate | Drop-connect Rate | Regularisation      | Accuracy |
|---------------------|----------------------|-----------------|--------------|-------------------|---------------------|----------|
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropout        | 93.9%    |
| Swish               | DL                   | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropout        | 91.9%    |
| Swish               | 0.1 * DL + 0.9 * BCE | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropout        | 93.5%    |
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropblock (2D) | 94.4%    |

DL - Dice Loss, BCE - Binary Crossentropy, SD - Stochastic Depth
