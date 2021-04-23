# OPU_Simulator

* fsim directory includes the functional simulator for OPU instruction sequence.

You can check [accelerator.cc](https://github.com/OPU-Lab/209-Project/blob/main/Simulator/fsim/accelerator.cc) and [accelerator.h](https://github.com/OPU-Lab/209-Project/blob/main/Simulator/fsim/accelerator.h) for architecture definition.

## Tiny YOLO test
```
mkdir workspace
cd workspace
pytest -v ../test/utest.py -k test_tiny_yolo
```
