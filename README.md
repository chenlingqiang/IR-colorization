# IR-colorization


We provide the Pytorch implementation of "Colorization of infrared images based on feature fusion and contrastive learning"
# Getting Started
## Installation
This code was tested with Pytorch 1.6.0, CUDA 10.2, and Python 3.8
## Testing
* Please download the pre-trained model and test set from [google drive link](https://drive.google.com/file/d/1yFLy_NUMF8jZfjQg-1LKLcOL-hiocVCi/view?usp=sharing), and put model under ./checkpoints/experiment_name/ . put test set under ./datasets/cyclegan/
* test the model 
 ```python test.py --dataroot datasets/cyclegan --model COLOR ```
 # Acknowledge
 Our code is developed based on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation)

