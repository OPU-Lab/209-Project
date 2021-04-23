In general, the arithmetic unit module performs p = x * y, for more than one multipliers, y is shared;

Compact different number of low precision multiplications into one dsp.
One DSP48E1 is configured as (A+D)*B+C
We can assign different A,B,C,D to have:      mixed precision, support both 2 int8 * int8 and 4 int4 * int3

For int8 * int8, x is [x0(int8), x1(int8)], and for int4 * int3, 
