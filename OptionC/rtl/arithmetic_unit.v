`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// UCLA EDA LAB
// -----------------------------------------------------------------------------
// Engineer       : Chen Wu
// Design Name    : opu series
// Module Name    : arithmetic_unit
// Target Devices : 325t
// Tool Versions  : Vivado 2020.1, Modelsim 2019.4
// Description    : 
//    In general, it performs p = x * y, for more than one multipliers, 
//    y is shared;
//
//    Compact different number of low precision multiplications into one dsp.
//    one DSP48E1 is configured as (A+D)*B+C, assign different A,B,C,D to have:
//      mixed precision, support both 2 int8*int8 and 4 int4*int3
//    
//    For int8*int8, x is [x0(int8), x1(int8)], and for int4*int3, 
//
//    Delay: 4 cycles
// Revision       :
// Version        Date        Author        Descriptin
// 1.0            2021-04-22  Chen Wu       Initial version
// -----------------------------------------------------------------------------


module arithmetic_unit (
  output  wire    [31 : 0]      p                         ,

  input           [15 : 0]      x                         ,
  input           [ 7 : 0]      y                         ,
  input                         mode                      ,

  input                         clk                       ,
  input                         reset                     
  );

  reg             [24 : 0]      dsp_a = 0                 ;
  reg             [17 : 0]      dsp_b = 0                 ;
  reg             [47 : 0]      dsp_c = 0                 ;
  reg             [24 : 0]      dsp_d = 0                 ;
  wire            [47 : 0]      dsp_p                     ;

  wire            [ 6 : 0]      dsp_a_part0               ;
  wire            [ 6 : 0]      dsp_a_part1               ;
  wire            [ 6 : 0]      dsp_a_part2               ;

  reg             [ 2 : 0]      dsp_c_flag  = 0           ;
  reg             [47 : 0]      dsp_c_nd    = 0           ;

  assign dsp_a_part0  = $signed(x[3 : 0])                                     ;
  assign dsp_a_part1  = {{(3){x[7]}}, x[7 : 4]} - {6'h0, x[3]}                ;
  assign dsp_a_part2  = {{(3){x[11]}}, x[11 : 8]} - {6'h0, dsp_a_part1[6]}    ;

  always @(posedge clk) begin
    dsp_c_flag[0] <= (x[3] ^ y[3]) & (|x[3 : 0]) & (|y[3 : 0])                ;
    dsp_c_flag[1] <= (x[7] ^ y[3]) & (|x[7 : 4]) & (|y[3 : 0])              ;
    dsp_c_flag[2] <= (x[11] ^ y[3]) & (|x[11 : 8]) & (|y[3 : 0])            ;
  end

  always @(posedge clk) begin
    dsp_c_nd  <=  ((x[7] ^ y[7]) & 
                   (|x[7 : 0]) & 
                   (|y[7 : 0])) ? 48'h0000_0001_0000 : 48'h0          ;
  end

  always @(posedge clk) begin
    if ( mode )
      dsp_a <= $signed({dsp_a_part2, dsp_a_part1, dsp_a_part0})       ;
    else
      dsp_a <= $signed(x[7 : 0])                                      ;
  end

  always @(posedge clk) begin
    if ( mode )
      dsp_b <= $signed(y[3 : 0])                                      ;
    else
      dsp_b <= $signed(y)                                             ;
  end

  always @(posedge clk) begin
    if ( mode )
      dsp_d <= $signed(x[15 : 12]) << 21                              ;
    else
      dsp_d <= $signed(x[15 : 8]) << 16                               ;
  end

  always @(posedge clk) begin
    if ( mode ) 
      case ( dsp_c_flag )
      3'h0  : dsp_c <=  {26'h0, 22'h00_0000}    ;
      3'h1  : dsp_c <=  {26'h0, 22'h00_0080}    ;
      3'h2  : dsp_c <=  {26'h0, 22'h00_4000}    ;
      3'h3  : dsp_c <=  {26'h0, 22'h00_4080}    ;
      3'h4  : dsp_c <=  {26'h0, 22'h20_0000}    ;
      3'h5  : dsp_c <=  {26'h0, 22'h20_0080}    ;
      3'h6  : dsp_c <=  {26'h0, 22'h20_4000}    ;
      3'h7  : dsp_c <=  {26'h0, 22'h20_4080}    ;
      endcase
    else
      dsp_c     <=  dsp_c_nd                    ;
  end

  dsp_mac u0_dsp_mac (
    .CLK    ( clk         ),
    .SCLR   ( reset       ),
    .A      ( dsp_a       ),
    .B      ( dsp_b       ),
    .C      ( dsp_c       ),
    .D      ( dsp_d       ),
    .P      ( dsp_p       ) 
  );

  assign p = mode ? {dsp_p[27], dsp_p[27:21],
                     dsp_p[20], dsp_p[20:14],
                     dsp_p[13], dsp_p[13: 7],
                     dsp_p[ 6], dsp_p[ 6: 0]} :
                    dsp_p[31:0]                 ;

endmodule