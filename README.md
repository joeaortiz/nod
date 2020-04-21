## Unsupervised scene decomposition

Unsupervised decomposition of scene into 2 objects and background. Uses multiview 3D cue that objects transform separately when moving from one view to another. 

Model architecture: 
- CSWM style encoder
- GNN transition model to change viewpoint
- Broadcast decoder

### Setup

conda env create -f environment.yml


