**Detailed documentation about the dataflow for convolution on the accelerator will be released soon!**

### Introduction
This project can generate software reference files, FPGA simulation files, and on-board test files for OPU.

### Code Hierarchy

```
opu_ref
    -- data: Used to store data for generating reference files.
        -- network_name: Please replace "network_name" with the specific name of the network, such as "yolov3".
            -- ins: Used to store the instructions for simulation.
            -- onboard_ins: Used to store the instructions for on-board testing.
            -- weights: Used to store kernel and bias files (*.mat files).
            -- inp.mat: Input image for testing.
            -- ir.txt: intermediate representation file.
    -- results: Used to store all the results. It will be created automatically while running.
        -- network_name: Please replace "network name" with the specific name of the network, should be the same as that in "data" folder.
            -- layer_0: Used to store all the reference files.
                -- 4hw: Used to store the files for FPGA simulation.
    -- src: Used to store all the source files.
```

Pay attention: Please create the data folders and put the files manually, as larger files, such as weights, are not uploaded.

