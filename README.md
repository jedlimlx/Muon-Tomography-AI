# Computed Tomography AI

The purpose of this AI is to perform computed tomography using x-rays / muons / other particles. 
Data is synthetic and obtained from Geant4.

## Experimental Logs

Experiments conducted are logged below.

### Muons Dataset

40 Epochs

| Activation Function | Loss Function        | Blocks          | Dropout Rate | Drop-connect Rate | Regularisation          | Attention | Accuracy |
|---------------------|----------------------|-----------------|--------------|-------------------|-------------------------|-----------|----------|
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropout            | None      | 93.9%    |
| Swish               | DL                   | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropout            | None      | 91.9%    |
| Swish               | 0.1 * DL + 0.9 * BCE | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropout            | None      | 93.5%    |
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropblock (2D)     | None      | 94.4%    |
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropblock (2D, 3D) | None      | 94.2%    |
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.10         | 0.20              | SD + Dropblock (2D, 3D) | None      | 88.6%    |
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.025        | 0.20              | SD + Dropblock (2D, 3D) | None      | 94.0%    |
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.025        | 0.20              | SD + Dropblock (2D, 3D) | SE        | 94.4%    |
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.025        | 0.20              | SD + Dropblock (2D)     | SE        | 94.2%    |
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropblock (2D)     | SE        | 94.0%    |
| Swish               | BCE                  | [1, 2, 2, 2, 3] | 0.10         | 0.20              | SD + Dropblock (2D)     | SE        | 94.5%    |

100 Epochs, Early Stopping (Patience 10)

| Activation Function | Loss Function | Blocks          | Dropout Rate | Drop-connect Rate | Regularisation      | Attention | Gaussian Noise | Learning Rate | Accuracy |
|---------------------|---------------|-----------------|--------------|-------------------|---------------------|-----------|----------------|---------------|----------|
| Swish               | BCE           | [1, 2, 2, 2, 3] | 0.10         | 0.20              | SD + Dropblock (2D) | None      | 0.20           | 0.001         | 93.8%    |
| Swish               | BCE           | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropblock (2D) | None      | 0.20           | 0.001         | 94.9%    |
| Swish               | BCE           | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropblock (2D) | None      | 0.50           | 0.001         | 94.2%    |
| Swish               | BCE           | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropblock (2D) | None      | 0.10           | 0.001         | 94.3%    |
| Swish               | BCE           | [1, 2, 2, 2, 3] | 0.05         | 0.20              | SD + Dropblock (2D) | None      | 0.20           | 0.002         | 94.9%    |

DL - Dice Loss, BCE - Binary Crossentropy, SD - Stochastic Depth
