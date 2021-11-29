# Fast Neural Style Transfer using Pytorch Models

This repository implements Perceptual Losses for Real-Time Style Transfer and Super-Resolution paper by Justin Johnson, Alexandre Alahi, and Fei-Fei Li, using pretrained VGG models in Pytorch models. Model architecture and training methodology is same, but hyper parameters are different. 

Pretrained models being different the loss function and relative strengths of style loss and content loss needed to be recalibrated. The loss with pretrained models is very small, so scaling it by 1.0e5 works relatively well for different styles.

Was able to replicate results shown in the paper below with pretrained VGG models provided on the Original webpage of the project.

Was able to create similar results with Pretrained Network provided with Pytorch Pretrained model, with different hyper parameters.

## Python Prerequisites

- python3.6 >
- pytorch with cuda enabled.
- cuda 10.2 version
- skimage
- dominate
- copy 


## Usage

### Training

```bash
python train.py -lr 0.001 -epoch 2 -batch 6 -style 5 -alphatv 1 -alpha 200000
```

Above example is for style 5. More can be used.

### Testing/ Running on your own images
```bash
python test.py -style 5 -imageName <imagename> 
```

<imagename> should contain the path to the file as well.


## Some Issues

1. Dynamic range of outputs is more than inputs, some way to normalize that would make the outputs better.


## Styles used for experiments

1. Style 0 [Sample Outputs for the Style](results/style0.md)

<img src='styles/starry_night.jpg' height='225px'>

2. Style 1

<img src='styles/the_scream.jpg' height='225px'>

3. Style 2

<img src='styles/udnie.jpg' height='225px'>

4. Style 3
<img src='styles/wave.jpg' height='225px'>

5. Style 4 [Sample Outputs for the Style](results/style4.md)
<img src='styles/mosaic.jpg' height='225px'>

6. Style 5  [Sample Outputs for the Style](results/style5.md)
<img src='styles/la_muse.jpg' height='225px'>

7. Style 6  [Sample Outputs for the Style](results/style6.md)
<img src='styles/candy.jpg' height='225px'>

8. Style 7  [Sample Outputs for the Style](results/style7.md)
<img src='styles/composition_vii.jpg' height='225px'>

9. Style 8  [Sample Outputs for the Style](results/style8.md)
<img src='styles/SampleStyle-2.jpg' height='225px'>

10. Style 9 [Sample Outputs for the Style](results/style9.md)
<img src='styles/SampleStyle-1.jpg' height='225px'>

11. Style 10  [Sample Outputs for the Style](results/style10.md)
<img src='styles/SampleStyle-4.jpg' height='225px'>

## Future Experiments if Possible

1. It will proabably work with other Networks as well like ResNet, might be interesting to see those results.

