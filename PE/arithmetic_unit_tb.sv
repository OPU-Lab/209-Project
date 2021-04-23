`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// UCLA EDA LAB
// -----------------------------------------------------------------------------
// Engineer       : Chen Wu
// Design Name    : opu series
// Module Name    : arithmetric_unit_tb
// Target Devices : 325t
// Tool Versions  : Vivado 2020.1, Modelsim 2019.4
// Description    : 
//    Testbench for arithmetic_unit
// Revision       :
// Version        Date        Author        Descriptin
// 1.0            2021-04-04  Chen Wu       Initial version
// -----------------------------------------------------------------------------

module arithmetic_unit_tb();

  localparam    TEST_NUM  = 16                            ;

  localparam    CYCLE     = 5.0                           ; 

  wire      [31 : 0]        p           ;
  reg       [15 : 0]        x           ;
  reg       [ 7 : 0]        y           ;
  reg                       mode        ;

  reg                       clk         ;
  reg                       reset       ;

  initial begin
    clk       =   1'b1                  ;
    forever #(CYCLE / 2.0)
      clk     =   ~clk                  ;
  end

  task init;
    reset     =   1'b1                  ;
    x         =   0                     ;
    y         =   0                     ;
    mode      =   1'b0                  ;

    repeat (20) @(posedge clk)          ;
    reset     =   1'b0                  ;
  endtask

  task run (input run_mode);
    integer         cnt = 0             ;

    forever @(posedge clk) begin
      if ( cnt == TEST_NUM ) begin
        x       <=  0                   ;
        y       <=  0                   ;
        break                           ;
      end else begin
        x       <=  $random() % 65536   ;
        y       <=  $random() % 256     ;
        mode    <=  run_mode            ;
      end
    end
  endtask

  initial begin
    init()                              ;
    run(0)                              ;

    repeat(20) @(posedge clk)           ;
    $finish                             ;
  end

  arithmetic_unit u0_arithmetic_unit (
    .p            ( p                 ),

    .x            ( x                 ),
    .y            ( y                 ),
    .mode         ( mode              ),

    .clk          ( clk               ),
    .reset        ( reset             )
  );

endmodule