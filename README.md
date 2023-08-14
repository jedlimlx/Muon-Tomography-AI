# \mu-Net: ConvNeXt-Based U-Nets for Muon Tomography

Muon scattering tomography is an imaging technique that utilizes muons, typically originating from cosmic rays to 
image the interiors of objects. Since muons are highly penetrating, muon tomography can be used to investigate the 
internal composition of dense materials such as geological formations or archaeological structures. 
However, due to the low flux of cosmic ray muons at sea-level and the highly complex interactions that muons display 
when travelling through matter, existing reconstruction algorithms often suffer from low resolution and high noise. 
In this work, we develop a novel two-stage deep learning algorithm, $\mu$-Net, consisting of an MLP to predict the muon 
trajectory and a ConvNeXt-based U-Net to convert the scattering points into voxels. $\mu$-Net is trained on synthetic 
data generated by the Geant4 simulation package and we show that it outperforms existing reconstruction methods 
for muon tomography. It achieves a state-of-the-art performance of 17 PSNR at a dosage of 1024 muons, outperforming 
traditional reconstruction algorithms such as the point of closest approach algorithm and maximum likelihood and 
expectation maximisation algorithm. Furthermore, we find that our method is robust to various corruptions such as 
inaccuracies in the muon momentum or a limited detector resolution. We hope that this research will spark further 
investigations into the potential of deep learning to revolutionise this field. 

Our dataset can be found https://www.kaggle.com/datasets/tomandjerry2005/muons-scattering-dataset.

The weights of the best performing models of each dosage at released under GitHub Releases. To load the model, run
```python
from layers.agg_3d import Agg3D

model = Agg3D(
    **{
        'point_size': 3,
        'downward_convs': [1, 1, 2, 3, 5],
        'downward_filters': [8, 16, 64, 128, 256],
        'upward_convs': [4, 3, 2, 1],
        'upward_filters': [128, 64, 16, 8],
        'resolution': 64,
        'noise_level': 0,
        'threshold': 1e-3
    }
)
model.load_weights("path-to-weights.h5")
```
