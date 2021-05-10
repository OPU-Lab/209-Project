## Requirements

In option B, you will
* develop a software model to simulate dataflow of convolution on hardware, including
    * tiling (blocked computation) with numpy 
    * 4-bit intermediate feature maps, 4-bit weight, and 8-bit bias with numpy
* simulate ResNet20 model    

You may assume the capacity of the on-chip feature map buffer is 40(T<sup>h</sup>) × 40(T<sup>w</sup>) × 64(T<sup>c</sup>) elements.


## Provided Data and Files

Under SoftwareModel directory, you can find a software model for OPU architecture in 8-bit.

Under data directory, you can find 4-bit weight and 8-bit bias (signed fixed-point) for each layer in ResNet20 (check its model architecture at https://github.com/OPU-Lab/209-Project/blob/main/ResNet20_architecture.md).
The index (starting from 0) comes from the order of linearized topological order of layers.
For example, weight_0.npy is the weight for the 1st conv2d in ResNet20.
For each basicblock with conv2d on the residual path, that conv2d is indexed before all the other layers in the same basicblock.
You can find fraction length information in [data/info.json](https://github.com/OPU-Lab/209-Project/blob/main/OptionB/data/info.json).
An example to find fraction length for each parameter is shown below.
```
import json

with open(path for info.json) as json_file: 
   lines = json_file.readlines()
   for i in range(len(lines)):
      line = lines[i]
      info = json.loads(line)
      # For ith layer
      ifmap_fl = info['input_fraclen']
      ofmap_fl = info['output_fraclen']
      weight_fl = info['weight_fraclen']
      bias_fl = info['bias_fraclen']
```
