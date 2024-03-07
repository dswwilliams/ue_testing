
### Overview

Repo for testing the quality of a segmentation network's uncertainty estimation.

Used the following projects:

[“Mitigating Distributional Shift in Semantic Segmentation via Uncertainty Estimation from Unlabelled Data”, D. Williams, D. De Martini, M. Gadd, and P. Newman, IEEE Transactions on Robotics (T-RO), 2024](https://arxiv.org/abs/2402.17653)
<!-- TODO: link this to my personal website not arxiv -->
```
@article{gammassl,
title={{Mitigating Distributional Shift in Semantic Segmentation via Uncertainty Estimation from Unlabelled Data}},
author={Williams, David and De Martini, Daniele and Gadd, Matthew and Newman, Paul},
booktitle={IEEE Transactions on Robotics (T-RO)},
year={2024},
}
```

[“Masked Gamma-SSL: Learning Uncertainty Estimation via Masked Image Modeling”, D. Williams, M. Gadd, P. Newman, and D. De Martini, IEEE International Conference on Robotics and Automation (ICRA), 2024](https://arxiv.org/abs/2402.17622)
<!-- TODO: link this to my personal website not arxiv -->
```
@article{maskedgammassl,
title={{Masked Gamma-SSL: Learning Uncertainty Estimation via Masked Image Modeling}},
author={Williams, David and Gadd, Matthew and Newman, Paul and De Martini, Daniele},
booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
year={2024},
}
```

### Environment
The required conda environment can be setup with:
```
conda env create -f environment.yml
conda activate gammassl
```