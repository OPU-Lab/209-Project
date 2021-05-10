## Requirements

In option C, you will 
* build a PE with 256 DSP Macros in RTL code
* decompose one DSP Macro for multiple low-precision multiplications
* simulate 1st conv2d of ResNet20 with the PE in three precisions: 8-bit, 4-bit, 2-bit

## Background

A PE consists of a bunch of parallel multipliers followed by adders for reduction.

Let's consider implement the PE with FPGA.
Assume we only use DSP Macros for multipliers. 
One DSP48E1 is configured as (A+D)xB+C, where A, B, C, D are 25, 18, 48, 25 bits and output is 48 bits.
For simplicity, if we set D and C to zeros, it performs a 25-bit x 18-bit multiplication. 
For low-precision data (e.g., 8-bit, 4-bit), one DSP can be decomposed into low-precision multipliers.

For example, we can use A for 2 int8 and B for 1 int8, where B is shared between the two at A.
Or we can use A for 4 int4 and B for 1 int3, where B is shared between the four at A.
As such, we can make full use of arithmetic resources on FPGA.
The above two example for decomposition have been realized at [https://github.com/OPU-Lab/209-Project/blob/main/OptionC/rtl/arithmetic_unit.v](https://github.com/OPU-Lab/209-Project/blob/main/OptionC/rtl/arithmetic_unit.v).

## Provided Data and Files
Under rtl directory, you can find
* arithmetic_unit.v: example decomposed DSP in Verilog HDL 
* dsp.xcix: a Xilinx IP core for DSP48E1. You are welcome to try with Xilinx toolchain.

Under data directory, you can find
* conv2d_input.npy: input tensor for the 1st conv2d in ResNet (full precision)
* conv2d_input_wl8_fl5.npy: input tensor for the 1st conv2d in ResNet (quantized, 8-bit signed fixed-point, fraction length = 5)
* conv2d_output_wl8_fl4.npy: output tensor for the 1st conv2d in ResNet (quantized, 8-bit signed fixed-point, fraction length = 4)
* weight_wl8_fl7.npy: weight for the 1st conv2d in ResNet (quantized, 8-bit signed fixed-point, fraction length = 7)
* weight_wl4_fl3.npy: weight for the 1st conv2d in ResNet (quantized, 4-bit signed fixed-point, fraction length = 3)
* weight_wl2_fl2.npy: weight for the 1st conv2d in ResNet (quantized, 2-bit signed fixed-point, fraction length = 2)
* sim_pe.npy: you can use it to generate input and output data for your testbench

For 8-bit, 4-bit, 2-bit precision, we all quantize ifmap to 5-bit fraction length, and ofmap to 4-bit fraction length. 
