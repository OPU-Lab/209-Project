## Requirements

In option B, you will
* develop a software model to simulate dataflow of convolution on hardware, including
    * tiling (blocked computation) with numpy 
    * 4-bit intermediate feature maps, 4-bit weight, and 8-bit bias with numpy
* simulate ResNet20 model    

You may assume the capacity of the on-chip feature map buffer is 40(T<sup>h</sup>) × 40(T<sup>w</sup>) × 64(T<sup>c</sup>) elements.


## Provided Data and Files

Under SoftwareModel directory, you can find a software model for OPU architecture in 8-bit.

Under data directory, you can find weight and bias for each layer in ResNet20 (check its model architecture at https://github.com/OPU-Lab/209-Project/blob/main/ResNet20_architecture.md).
The index (starting from 0) comes from the order of linearized topological order of layers.
For each basicblock with conv2d on the residual path, that conv2d is indexed before all the other layers in the same basicblock.
