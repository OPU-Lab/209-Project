create_ip -module_name dsp_mac -vendor xilinx.com -library ip -name xbip_dsp48_macro

set_property -dict [list \
  CONFIG.instruction1 {(A+D)*B+C} \
  CONFIG.pipeline_options {Expert} \
  CONFIG.dreg_3 {false} \
  CONFIG.areg_3 {false} \
  CONFIG.breg_3 {false} \
  CONFIG.creg_3 {false} \
  CONFIG.d_width {25} \
  CONFIG.a_width {25} \
  CONFIG.b_width {18} \
  CONFIG.c_width {48} \
  CONFIG.has_sclr {true} \
  CONFIG.areg_4 {true} \
  CONFIG.breg_4 {true} \
  CONFIG.creg_4 {true} \
  CONFIG.creg_5 {true} \
  CONFIG.mreg_5 {true} \
  CONFIG.preg_6 {true} \
  CONFIG.d_binarywidth {0} \
  CONFIG.a_binarywidth {0} \
  CONFIG.b_binarywidth {0} \
  CONFIG.concat_width {48} \
  CONFIG.concat_binarywidth {0} \
  CONFIG.c_binarywidth {0} \
  CONFIG.pcin_binarywidth {0} \
  CONFIG.p_full_width {48} \
  CONFIG.p_width {48} \
  CONFIG.p_binarywidth {0} \
] [get_ips dsp_mac]
