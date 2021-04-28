# opu-fsim

<!-- ![simulation diagram](https://github.com/ucla-eda-lab-fpga/OPU_Simulator/blob/master/fsim/fsim.png) -->

### Build
```
mkdir build
cd build
cmake ..
make -j4
```

### Examples

* tiny yolo

Download binary input feature map, weight, bias and instructions from [Google Drive](https://drive.google.com/drive/folders/1XymCCD3ZlDk49-zpPxto3Bk5WLV2j69A?usp=sharing) and place them under `example/tinyyolo/`.
Change the first line of [config.h](https://github.com/ucla-eda-lab-fpga/OPU_Simulator/edit/master/fsim/config.h) to `#define TINYYOLO`.
Then run `../build/simulator` under `example/`.

* yolov3

Download binaries from [Google Drive](https://drive.google.com/drive/folders/1lFPuku4eIVQmzDuL0GW-izwC8M5iT4-Y?usp=sharing) and place them under `example/yolov3/`.
Change the first line of [config.h](https://github.com/ucla-eda-lab-fpga/OPU_Simulator/edit/master/fsim/config.h) to `#define YOLOV3`.
Then run `../build/simulator` under `example/`.


### Utility

* txt2bin: convert instruction in text format to binary, which can be used as input ins.bin for simulator
```
./txt2bin input_binary.txt output.bin
```
* reshape_input.py: reshape input feature map
```
python3 reshape_input.py --input input.npy --padding 0 0 0 0 --kz 1 5 --fl 5
```
