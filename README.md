# GANs-pytorch
 Collection of GAN models and implementations

### Basic GAN
Basic GAN implementation using MNIST 
#### Summary
 Using a relatively simple GAN model. </br>
 
 MNIST data can be found [here](http://yann.lecun.com/exdb/mnist/) or in this case from [torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
 #### Findings
 GAN model trained on 200 epochs. <br>
 
| 1 epoch     | 50 epochs | 100 epochs  |  150 epochs     |   200 epochs   |
| ------------|:---------:|:-----:|:-----:|:-----:|
| <img src ="GAN/gan_images/1.png" width = 200>   | <img src ="GAN/gan_images/50.png" width = 200> | <img src ="GAN/gan_images/100.png" width = 200> | <img src ="GAN/gan_images/150.png" width = 200> | <img src ="GAN/gan_images/200.png" width = 200> |
<br>

### DCGAN
Deep-Convolutional GAN implementation using CelebA 
#### Summary
 Using a Deep-Convolutional GAN model. </br>
 
 CelebA data can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
 #### Findings
 GAN model trained on 5 epochs. <br>
 
 <img src ="DCGAN/gan_images/gan_5e.gif" width = 400>
 
 Result after 5 epochs
 
 <img src ="DCGAN/gan_images/gan_gen_5e.png" width = 400>
