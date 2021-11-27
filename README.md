# Fast Neural Style Transfer using Pytorch Models

This repository implements Perceptual Losses for Real-Time Style Transfer and Super-Resolution paper by Justin Johnson, Alexandre Alahi, and Fei-Fei Li, using pretrained VGG models in Pytorch models. Model architecture and training methodology is same, but hyper parameters are different. 

Pretrained models being different the loss function and relative strengths of style loss and content loss needed to be recalibrated. The loss with pretrained models is very small, so scaling it by 1.0e5 works relatively well for different styles.

Was able to replicate results shown in the paper below with pretrained VGG models provided on the Original webpage of the project.

Was able to create similar results with Pretrained Network provided with Pytorch Pretrained model, with different hyper parameters.

1. It will proabably work with other Networks as well like ResNet, might be interesting to see those results.

## Styles used for experiments

1. Style 0

<img src='styles/starry_night.jpg' height='225px'>

2. Style 1

<img src='styles/the_scream.jpg' height='225px'>

3. Style 2

<img src='styles/udnie.jpg' height='225px'>

4. Style 3
<img src='styles/wave.jpg' height='225px'>

5. Style 4
<img src='styles/mosaic.jpg' height='225px'>

6. Style 5
<img src='styles/la_muse.jpg' height='225px'>

7. Style 6
<img src='styles/candy.jpg' height='225px'>

8. Style 7
<img src='styles/composition_vii.jpg' height='225px'>

9. Style 8
<img src='styles/SampleStyle-2.jpg' height='225px'>

10. Style 9
<img src='styles/SampleStyle-1.jpg' height='225px'>

11. Style 10
<img src='styles/SampleStyle-4.jpg' height='225px'>

