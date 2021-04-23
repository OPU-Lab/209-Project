#include "./accelerator.h"
#include <glog/logging.h>

/*
 * Function wrapper for DDR load instruction, which tarverses
 * load options and perform memory transaction from DDR to SRAM
 * accordingly for feature map, weight, bias, instruction
 */
void Device::RunLoad(OPUDDRLDInsn* load) {
  std::cout << "*************DDR LD**************\n";
  load_single = load->ddr_load_single;
  for (auto mem_id : load->ddr_load_type) {
    std::cout << mem_id << "\n";
    if (mem_id == OPU_MEM_ID_FM) {  // load feature map
      Fm_ram_t* fm_ram = fm_ram_vec_[fm_ram_id];
      std::cout << "LD to fm[" << fm_ram_id <<"]\n";
      fm_ram_id = (fm_ram_id + 1) % fm_ram_vec_.size();
      for (int i = 0; i < load->ddr_load_block_y_size; i++) {
        size_t src_addr = load->ddr_fm_addr_ini + i * load->fm_in_x_size;
        size_t dst_addr = i * load->ddr_load_block_x_size;
        MemOp mem = MemOp(
            src_addr,
            load->ddr_load_block_x_size,
            dst_addr);
        fm_ram->Load<512>(mem, dram.GetAddr(mem.src_addr));
      }
      std::cout << load->ddr_fm_addr_ini << "[" << load->ddr_load_block_y_size
                << " x " << load->ddr_load_block_x_size <<"]\n";
    } else if (mem_id == OPU_MEM_ID_WGT) {  // load weight
      Wgt_ram_t* wgt_ram = wgt_ram_vec_[wgt_ram_id];
      std::cout << "LD to wgt[" << wgt_ram_id <<"]\n";
      wgt_ram_id = (wgt_ram_id + 1) % wgt_ram_vec_.size();
      MemOp mem = MemOp(
            load->ddr_ker_addr_ini,
            load->ddr_ker_read_num,
            0);
      wgt_ram->Load<512>(mem, dram.GetAddr(mem.src_addr));
      std::cout << load->ddr_ker_addr_ini << "[" << load->ddr_ker_read_num << "]\n";
    } else if (mem_id == OPU_MEM_ID_BIAS) {  // load bias
      Bias_ram_t* bias_ram = bias_ram_vec_[bias_ram_id];
      std::cout << "LD to bias[" << bias_ram_id <<"]\n";
      bias_ram_id = (bias_ram_id + 1) % bias_ram_vec_.size();
      MemOp mem = MemOp(
            load->ddr_bias_addr_ini,
            load->ddr_bias_read_num,
            0);
      bias_ram->Load<512>(mem, dram.GetAddr(mem.src_addr));
      std::cout << load->ddr_bias_addr_ini << "[" << load->ddr_bias_read_num << "]\n";
    } else if (mem_id == OPU_MEM_ID_RESIDUAL) {  // load feature map for residual add

    } else if (mem_id == OPU_MEM_ID_INS) {  // load instruction

    } else {
      std::cout << "[ERROR] unknown ddr load type:" << mem_id << "\n";
      exit(1);
    }
  }
  
  // cycle count
  int64_t ddr_init_lat = 20;
  int64_t ddr_burst_length = 8;
  profiler.collect(0);
  for (auto mem_id : load->ddr_load_type) {
    if (mem_id == OPU_MEM_ID_FM) {
      for (int i = 0; i < load->ddr_load_block_y_size; i++) {
        profiler.incrementLast(
            static_cast<int64_t>(
            std::ceil(static_cast<double>(load->ddr_load_block_x_size)
            / static_cast<double>(ddr_burst_length)))
            + ddr_init_lat);
      }
    } else if (mem_id == OPU_MEM_ID_WGT) {
      profiler.incrementLast(
        static_cast<int64_t>(
        std::ceil(static_cast<double>(load->ddr_ker_read_num)
        / static_cast<double>(ddr_burst_length)))
        + ddr_init_lat);
    } else if (mem_id == OPU_MEM_ID_BIAS) {
      profiler.incrementLast(
        static_cast<int64_t>(
        std::ceil(static_cast<double>(load->ddr_bias_read_num)
        / static_cast<double>(ddr_burst_length)))
        + ddr_init_lat);
    }
  }
}

/*
 * Function wrapper for DDR store instruction, which applies post CONV processings,
 * including activation, pooling, residual addition, padding, (upsampling) in the
 * order dynamically spefcified by instructions
 */
void Device::RunStore(OPUDDRSTInsn* store) {
  std::cout << "*************DDR ST**************\n";
  std::cout << "ddr_save_pos = " << store->ddr_save_pos << "\n";
  std::cout << "residual = " << store->residual << "\n";
  if (store->residual)
    std::cout << "fm[" << ((compute_ins_->fm_ram_id + fm_ram_vec_.size() - 1)
    % fm_ram_vec_.size()) << "]\n";
  std::cout << "activation_enable = " << store->activation
            << " type = " << store->activation_type << "\n";
  std::cout << "pooling_enable = " << store->pooling
            << " type = " << store->pooling_type << "\n";
  std::cout << "pooling_x_size = " << store->pooling_x_size
            << "(" << store->pooling_x_stride << ")\n";
  std::cout << "pooling_y_size = " << store->pooling_y_size
            << "(" << store->pooling_y_stride << ")\n";
  std::cout << "ST to ddr addr ini = " << store->fm_output_addr_ini << "\n";
  std::cout << "ddr_save_fm_num = " << store->ddr_save_fm_num << "\n";
  std::cout << "ddr_save_block_x_size = "
            << store->ddr_save_block_x_size << "\n";
  std::cout << "ddr_save_block_y_size = "
            << store->ddr_save_block_y_size << "\n";
  std::cout << "fm_out_x_size = " << store->fm_out_x_size << "\n";
  std::cout << "fm_out_y_size = " << store->fm_out_y_size << "\n";
  std::cout << "padding = " << store->padding << "\n";
  std::cout << "padding_size = " << store->padding_size << "\n";
  std::cout << "upsampling = " << store->upsample_output << "\n";
  
  if (skip_exec) return;
  
  if (store->padding == 1) {
    // Padding
    RunPadding(store);
  } else {
    // Post-processing
    RunPostProcess(store);
  }

  // Debug
  if (store->padding == 1) {
    std::ofstream os_fm("ofm.txt");
    for (int i = 0; i < store->fm_out_x_size * store->fm_out_y_size * std::ceil(store->channel_out / 64.0); i++) {
      int addr = store->fm_output_addr_ini + i;
      std::vector<int8_t> res8(64);
      void* data = dram.GetAddr(addr);
      std::memcpy(&res8[0], data, 64);
      std::vector<int> res;
      for (auto x : res8) {
        res.push_back(static_cast<int>(x));
      }
      writeOut<int, 2>(os_fm, res, true);
    }
    os_fm.close();
  }
}

/*
 * Padding output feature map with zeros
 */
void Device::RunPadding(OPUDDRSTInsn* store) {
  std::vector<int8_t> res8(64, 0);
  int padding_size_l = store->padding_size;
  int padding_size_r = store->padding_size;
  int padding_size_u = store->padding_size;
  int padding_size_d = store->padding_size;
  int pad_cnt = 0;
  // Data layout :
  //   (row major) fm_x*fm_y*channel[63:0] - fm_x*fm_y*channel[127:64]
  for (int p = 0; p < store->channel_out; p += 64) {
    int addr = store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size;
    // Top
    for (int i = 0; i < padding_size_u; i++) {
      for (int j = 0; j < store->fm_out_x_size; j++) {
        dram.Write(addr++, &res8[0], 64);
        pad_cnt++;
      }
    }
    CHECK_EQ(addr, store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size
        + padding_size_u * store->fm_out_x_size);
    // Two sides
    for (int i = 0;
        i < store->fm_out_y_size - padding_size_u - padding_size_d; i++) {
      addr = store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size
        + (padding_size_u + i) * store->fm_out_x_size;
      for (int j = 0; j < padding_size_l; j++) {
        dram.Write(addr + j, &res8[0], 64);
        pad_cnt++;
      }
      for (int j = 0; j < padding_size_r; j++) {
        dram.Write(addr + store->fm_out_x_size - padding_size_r + j,
            &res8[0], 64);
        pad_cnt++;
      }
    }
    // Bottom
    addr = store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size
        + (store->fm_out_y_size - padding_size_d) * store->fm_out_x_size;
    for (int i = 0; i < padding_size_d; i++) {
      for (int j = 0; j < store->fm_out_x_size; j++) {
        dram.Write(addr++, &res8[0], 64);
        pad_cnt++;
      }
    }
    CHECK_EQ(addr, store->fm_output_addr_ini
        + p * store->fm_out_y_size * store->fm_out_x_size
        + store->fm_out_y_size * store->fm_out_x_size);
  }
  // Check total padding count
  CHECK_EQ(pad_cnt, store->ddr_save_fm_num);
  profiler.collect(pad_cnt);
}

/*
 * Post-processing according to pooling's memory access pattern
 */
void Device::RunPostProcess(OPUDDRSTInsn* store) {
  int64_t initiate_cnt = 0;
  std::vector<int> res(64);
  // fetch ofm buffer according to pooling pattern
  for (int i = 0;
        i < store->ddr_save_block_y_size - store->pooling_y_size + 1;
        i += store->pooling_y_stride) {
    for (int j = 0;
        j < store->ddr_save_block_x_size - store->pooling_x_size + 1;
        j += store->pooling_x_stride) {
      for (int y = 0; y < store->pooling_y_size; y++) {
        for (int x = 0; x < store->pooling_x_size; x++) {
          initiate_cnt++;
          int addr = (i+y) * store->ddr_save_block_x_size + j + x;
          std::vector<int> temp = tmp_buf_.AsVec(addr, 64, 16);
          if (store->ddr_save_pos == 3) {
            Activation(temp, store->activation_type);
            Pooling(res, temp, store->pooling_type, x == 0 && y == 0);
          } else if (store->ddr_save_pos == 2) {
            Activation(temp, store->activation_type);
            Pooling(res, temp, store->pooling_type, x == 0 && y == 0);
            ResidualAdd(res, store->residual, addr);
          } else if (store->ddr_save_pos == 1) {
            Activation(temp, store->activation_type);
            ResidualAdd(temp, store->residual, addr);
            Pooling(res, temp, store->pooling_type, x == 0 && y == 0);
          } else {
            ResidualAdd(temp, store->residual, addr);
            Activation(temp, store->activation_type);
            Pooling(res, temp, store->pooling_type, x == 0 && y == 0);
          }
        }
      }
      std::vector<int8_t> res8;
      for (auto x : res) {
        res8.push_back(static_cast<int8_t>(x & 0xFF));
      }
      // Upsampling
      // TODO : scale currently fixed as 2
      int upsampling_scale = store->upsample_output? 2 : 1;
      for (int y = 0; y < upsampling_scale; y++) {
        for (int x = 0; x < upsampling_scale; x++) {
        #ifdef DEBUG_POST_OUT
          writeOut<int, 2>(os, res, dump);
        #endif
          // Write results to DDR
          int coord_y = i / store->pooling_y_stride * upsampling_scale + y;
          int coord_x = j / store->pooling_x_stride * upsampling_scale + x;
          int ddr_addr = store->fm_output_addr_ini +
                coord_y * store->fm_out_x_size +
                coord_x;
          dram.Write(ddr_addr, &res8[0], 64);
        }
      }
    }
  }
  
  int64_t ii = 1;
  int64_t epilogue = 20;
  int64_t latency = initiate_cnt * ii + epilogue;
  profiler.collect(latency);//epilogue);
  /*if (reg_.dw_flag)
    profiler.collect(epilogue);
  else
    profiler.collect(latency);*/
}

/*
 * Residual Add
 */
void Device::ResidualAdd(std::vector<int>& data, bool enable, int addr) {
  if (enable) {
    // Use the fm ram other than the one for compute
    // TODO : uniform buffer arbitration!
    int rid = (compute_ins_->fm_ram_id + fm_ram_vec_.size() - 1)
        % fm_ram_vec_.size();
    std::vector<int> residue =
        fm_ram_vec_[rid]->AsVec(addr, 512, 8);
    for (int i = 0; i < 64; i++) {
      data[i] += residue[i];
      data[i] = Saturate(data[i], data[i] > 0, 8);
    }
  } 
}

/*
 * Activation
 */
void Device::Activation(std::vector<int>& data, int type) {
  if (type == NO_RELU_CODE || reg_.activation == 0) {
    return;
  } else if (type == RELU_CODE) {  // relu
    for (auto& item : data) {
      item = (item < 0)? 0 : item;
    }
  } else if (type == LEAKY_RELU_CODE) {  // leaky_relu: *0.125 == >>3 and round
    for (auto& item : data) {
      if (item < 0) {
        if (item & 0x4) {
          item = (item >> 3) + 1;
        } else {
          item = item >> 3;
        }
      }
    }
  }
}

/*
 * Pooling
 */
void Device::Pooling
  (std::vector<int>& data_o, std::vector<int> data_i, int type, bool st) {
  auto it = data_o.begin();
  auto ie = data_i.begin();
  if (type == 0 || reg_.pooling == 0) {
    data_o.clear();
    data_o.insert(data_o.begin(), data_i.begin(), data_i.end());
  } else if (type == 1) {  // max
    while (it != data_o.end()) {
      if (st || *ie > *it) {
        *it = *ie;
      }
      it++;
      ie++;
    }
  } else {
    data_o.clear();
    data_o.insert(data_o.begin(), data_i.begin(), data_i.end());
  }
}

/*
 * Function wrapper for inner product compuation
 */
void Device::RunCompute(OPUComputeInsn* compute) {
  std::cout << "*************Compute**************\n";
  Fm_ram_t* fm_ram = fm_ram_vec_[compute->fm_ram_id];
  std::cout << "Fetch from fm[" << compute->fm_ram_id <<"]\n";
  Wgt_ram_t* wgt_ram = wgt_ram_vec_[compute->wgt_ram_id];
  std::cout << "Fetch from wgt[" << compute->wgt_ram_id <<"]\n";
  Bias_ram_t* bias_ram = bias_ram_vec_[compute->bias_ram_id];
  std::cout << "Fetch from bias[" << compute->bias_ram_id <<"]\n";
  std::cout << "fm y[" << compute->dma_y_min << ":" << compute->dma_y_max << "]"
            << "x[" << compute->dma_x_min << ":" << compute->dma_x_max << "]\n";
  std::cout << "final_output = " << compute->final_output << "\n";
  if (compute->dw_flag) {
    std::cout << "[DW]\n";
    RunComputeDW(compute);
    return;
  }
  
  if (compute->type == 0) {
    std::cout << "[FC]\n";
    RunComputeFC(compute);
    return;
  }
  
  // Instrumentation
  int64_t initiate_cnt = 0;
  // Control flags
  compute_finish = compute->final_output? true : false;
  compute_cnt++;
  
  if (skip_exec) return;
  
  // Load bias first
  ipa_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0), bias_ram->BeginPtr(0));
  std::vector<int> bias = ipa_.adder_buf_b_.AsVec(0, 64, 16);
  // Fetch data from SRAMs to IPA's compute buffers and then compute
  int k = 0;
  int wgt_addr = compute->ker_addr_s;
  int tmp_addr = 0;
  for (int i = compute->dma_y_min; i <= compute->dma_y_max;
           i += compute->read_y_stride) {
    for (int j = compute->dma_x_min; j <= compute->dma_x_max;
           j += compute->read_x_stride) {
      for (int p = 0; p < compute->ker_round; p++) {
        initiate_cnt++;
        // Get fm addr
        int fm_addr = i * compute->dma_block_x_size + j;
        int pe_bytes = sizeof(IPA_t::PE_buf_t::DType);  // 512 (* 8b)
        int fm_bytes = sizeof(Fm_ram_t::DType);  // 64 (* 8b)
        
        // Load data from sram to ipa rams
        // Replicate fm -> 512 * 8b
        int rep_num = 8 << compute->copy_mode;
        int valid_bytes = pe_bytes / rep_num;
        int8_t* fm = new int8_t[pe_bytes];
        for (int ii = 0; ii < rep_num; ii++) {
          int8_t* src = reinterpret_cast<int8_t*>(fm_ram->BeginPtr(fm_addr));
          std::memcpy(fm+ii*valid_bytes, src + fm_bytes - valid_bytes,
            valid_bytes);
        }
        ipa_.fm_buf_.Load<4096>(MemOp(0, 1, 0), fm);
        delete [] fm;
    #ifdef DEBUG_DMA_FM
        std::vector<int> gdb = ipa_.fm_buf_.AsVec(0, 512, 8);
        writeOut<int, 2>(os, gdb, dump);
    #endif
        
        // Get wgt addr
        wgt_addr = compute->ker_addr_s + p;
        // Split ker
        int8_t* wgt_src_a = reinterpret_cast<int8_t*>(
            ipa_.wgt_buf_a_.BeginPtr(0));
        int8_t* wgt_src_b = reinterpret_cast<int8_t*>(
            ipa_.wgt_buf_b_.BeginPtr(0));
        int8_t* wgt = reinterpret_cast<int8_t*>(wgt_ram->BeginPtr(wgt_addr));
        for (int ii = 0; ii < pe_bytes; ii++) {
          wgt_src_a[ii] = wgt[2 * ii];
          wgt_src_b[ii] = wgt[2 * ii + 1];
        }
    #ifdef DEBUG_DMA_KER
        std::vector<int> gdb = wgt_ram->AsVec(wgt_addr, 1024, 8);
        writeOut<int, 2>(os, gdb, dump);
    #endif
        // Compute
        ipa_.Forward(2 << compute->output_num);

        if (ipa_.GetOutputNum() >= compute->output_channel) {
          // Load out_adder_b
          //   - bias : loaded at the beginning
          //   - temp : partial sum from tmp_buf_
          ipa_.adder_b.clear();
          if (compute->add_bias) {
            for (auto value : bias) {
              ipa_.adder_b.push_back(
                  Saturate(value << compute->shift_num_bias, value > 0, 26));
            }
          } else if (compute->add_temp) {
            ipa_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0),
                tmp_buf_.BeginPtr(tmp_addr));
            std::vector<int> tmp = ipa_.adder_buf_b_.AsVec(0, 64, 16);
            for (auto value : tmp) {
              ipa_.adder_b.push_back(static_cast<int>(value) << 10);
            }
          }

          // Load adder_a from ipa output
          ipa_.Accumulate(
                      compute->shift_num_fm,
                      compute->final_output,
                      tmp_buf_.BeginPtr(tmp_addr), os, dump);
          tmp_addr++;
        }
      }
    }
  }
  
  // cycle count
  int64_t ii = 1;
  int64_t epilogue = 30;
  int64_t latency = initiate_cnt * ii + epilogue;
  profiler.collect(latency);
}

void Device::RunComputeFC(OPUComputeInsn* compute) {
  Fm_ram_t* fm_ram = fm_ram_vec_[compute->fm_ram_id];
  Wgt_ram_t* wgt_ram = wgt_ram_vec_[compute->wgt_ram_id];
  Bias_ram_t* bias_ram = bias_ram_vec_[compute->bias_ram_id];
  
  // Load bias first
  ipa_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0), bias_ram->BeginPtr(0));
  std::vector<int> bias = ipa_.adder_buf_b_.AsVec(0, 64, 16);
  
  // Control flags
  compute_finish = compute->final_output? true : false;
  compute_cnt++;
  
  int wgt_addr = compute->ker_addr_s;
  int tmp_addr = fc_tmp_addr;
  for (int y = 0; y < 4; y++) {
    ipa_.psum.insert(ipa_.psum.begin(), 0);
  }
  for (int i = compute->dma_y_min; i <= compute->dma_y_max;
         i += compute->read_y_stride) {
    for (int j = compute->dma_x_min; j <= compute->dma_x_max;
         j += compute->read_x_stride) {
        // Get fm addr
        int fm_addr = i * compute->dma_block_x_size + j;
        int pe_bytes = sizeof(IPA_t::PE_buf_t::DType);  // 512 (* 8b)
        int fm_bytes = sizeof(Fm_ram_t::DType);  // 64 (* 8b)
        
        int rep_num = 8 << compute->copy_mode;
        int valid_bytes = pe_bytes / rep_num;
        int8_t* fm = new int8_t[pe_bytes];
        for (int ii = 0; ii < rep_num; ii++) {
          int8_t* src = reinterpret_cast<int8_t*>(fm_ram->BeginPtr(fm_addr));
          std::memcpy(fm+ii*valid_bytes, src + fm_bytes - valid_bytes,
            valid_bytes);
        }
        ipa_.fm_buf_.Load<4096>(MemOp(0, 1, 0), fm);
        delete [] fm;
        /*std::cout << fm_addr << " " << valid_bytes << "\n";
        int8_t* src = reinterpret_cast<int8_t*>(fm_ram->BeginPtr(fm_addr));
        for (int i = 0; i < 64; i ++){
            std::cout << (int)src[i] << " ";
        }
        std::cout << "\n";*/
    #ifdef DEBUG_DMA_FM
        std::vector<int> gdb = ipa_.fm_buf_.AsVec(0, 512, 8);
        writeOut<int, 2>(os, gdb, dump);
    #endif
        
        // Get wgt addr
        wgt_addr = compute->ker_addr_s + static_cast<int>(std::ceil(fm_addr/ 4));
        // Split ker
        int8_t* wgt_src_a = reinterpret_cast<int8_t*>(
            ipa_.wgt_buf_a_.BeginPtr(0));
        int8_t* wgt_src_b = reinterpret_cast<int8_t*>(
            ipa_.wgt_buf_b_.BeginPtr(0));
        int8_t* wgt = reinterpret_cast<int8_t*>(wgt_ram->BeginPtr(wgt_addr));
        int offset = (fm_addr % 4) * fm_bytes * 2 * 2;
        for (int ii = 2 * fm_bytes; ii < pe_bytes; ii++) {
          wgt_src_a[ii] = 0;
          wgt_src_b[ii] = 0;
        }
        for (int ii = 0; ii < fm_bytes * 2; ii++) {
          wgt_src_a[ii] = wgt[2 * ii + offset];
          wgt_src_b[ii] = wgt[2 * ii + 1 + offset];
        }
    #ifdef DEBUG_DMA_KER
        std::vector<int> gdb = wgt_ram->AsVec(wgt_addr, 1024, 8);
        writeOut<int, 2>(os, gdb, dump);
    #endif
        
        // Compute
        ipa_.Forward(16);
        ipa_.psum.erase(ipa_.psum.begin() + 4, ipa_.psum.begin() + 16);
        // FC accumulation  
        for (int y = 0; y < 4; y++) {
          ipa_.psum[y] += ipa_.psum[y + 4];
        }
        ipa_.psum.erase(ipa_.psum.begin() + 4, ipa_.psum.begin() + 8);
    }
  }
    std::cout << ipa_.GetOutputNum() << " v.s " << compute->output_channel << "\n";
    if (ipa_.GetOutputNum() >= compute->output_channel) {
      // Load out_adder_b
      //   - bias : loaded at the beginning
      //   - temp : partial sum from tmp_buf_
      std::cout << "add_bias: " << compute->add_bias
                << " add_temp: " << compute->add_temp << "\n";
      ipa_.adder_b.clear();
      if (compute->add_bias) {
        for (auto value : bias) {
          ipa_.adder_b.push_back(
              Saturate(value << compute->shift_num_bias, value > 0, 26));
        }
      } else if (compute->add_temp) {
        ipa_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0),
            tmp_buf_.BeginPtr(tmp_addr));
        std::vector<int> tmp = ipa_.adder_buf_b_.AsVec(0, 64, 16);
        for (auto value : tmp) {
          ipa_.adder_b.push_back(static_cast<int>(value) << 10);
        }
      }

      // Load adder_a from ipa output
      ipa_.Accumulate(
                  compute->shift_num_fm,
                  compute->final_output,
                  tmp_buf_.BeginPtr(tmp_addr), os, dump);
      std::cout << "tmp buffer addr: " << fc_tmp_addr << "\n";
      fc_tmp_addr = ++tmp_addr;
      if (fc_tmp_addr == compute->output_block_y_size * compute->output_block_x_size) {
        fc_tmp_addr = 0;
      }
    }
  profiler.collect(1);
}  


void Device::RunComputeDW(OPUComputeInsn* compute) {
  Fm_ram_t* fm_ram = fm_ram_vec_[compute->fm_ram_id];
  Wgt_ram_t* wgt_ram = wgt_ram_vec_[compute->wgt_ram_id];
  Bias_ram_t* bias_ram = bias_ram_vec_[compute->bias_ram_id];
  // wgt_ram --> wgt_ram_dw_
  // 1024 * 8--> 2048 * 8 (64*32 filled from 64*18)
  int idx = 0;
  int tmp = 0;
  int8_t* wgt_dw = new int8_t[2048];
  for (int i = 0; i < 64; i++) {
    int8_t* wgt = reinterpret_cast<int8_t*>(wgt_ram->BeginPtr(i));
    for (int j = 0; j < 16; j++) {
      std::memcpy(wgt_dw + idx * 64, wgt + j * 64, 64);
      if (idx == 17) {
        std::memset(wgt_dw + 18 * 64, 0, 14*64);
        MemOp mem = MemOp(
            0,
            1,
            tmp++);
        wgt_ram_dw_.Load<2048>(mem, wgt_dw);
      }
      idx = (idx + 1) % 18;
    }
  }
  delete [] wgt_dw;
  // Instrumentation
  int64_t initiate_cnt = 0;
  // Control flags
  compute_finish = compute->final_output? true : false;
  compute_cnt++;
  
  if (skip_exec) return;
  
  // Load bias first
  ipa_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0), bias_ram->BeginPtr(0));
  std::vector<int> bias = ipa_.adder_buf_b_.AsVec(0, 64, 16);
  // Fetch data from SRAMs to IPA's compute buffers and then compute
  int k = 0;
  int wgt_addr = compute->ker_addr_s;
  int tmp_addr = 0;
  // To be more accurate, we should follow dma_min/max and write them to 64 parallel buffer
  // first. Then we fetch 3x3 data window from line buffer.
  // Here I skip the middle part, so use (y_max-ky+1) as dma access upper bound
  // not accurate but works for now for stride=1
  for (int i = compute->dma_y_min; i <= compute->dma_y_max - compute->ker_y_size + 1;
           i += compute->read_y_stride) {
    for (int j = compute->dma_x_min; j <= compute->dma_x_max - compute->ker_x_size + 1;
           j += compute->read_x_stride) {
      for (int p = 0; p < compute->ker_round; p++) {
        initiate_cnt++;
        continue;
        // Get fm addr
        int fm_addr_st = i * compute->dma_block_x_size + j;
        int pe_bytes = sizeof(IPA_t::PE_buf_t::DType);  // 16*64 (* 8b)
        int fm_bytes = sizeof(Fm_ram_t::DType);  // 64 (* 8b)
        
        // Load data from sram to ipa rams via line buffer
        int8_t* fm = new int8_t[pe_bytes];
        std::memset(fm, 0, pe_bytes);      
        int ker_cnt = compute->ker_y_size * compute->ker_x_size;
        for (int kh = 0; kh < compute->ker_y_size; kh++) {
          for (int kw = 0; kw < compute->ker_x_size; kw++) {
            int fm_addr = fm_addr_st + kh * compute->dma_block_x_size + kw;
            // split to fill 16 inputs of PE, 64 in total
            // for example, {9 elements, (7){0}} x 64 for ipa_.fm_buf_
            int8_t* src = reinterpret_cast<int8_t*>(fm_ram->BeginPtr(fm_addr));
            for (int ii = 0; ii < 64; ii++) {
              int byte_idx = ii * 16 + kh * compute->ker_x_size + kw;
              fm[byte_idx] = src[ii];
            }
          }
        }
        ipa_dw_.fm_buf_.Load<8192>(MemOp(0, 1, 0), fm);
        // select 9 from 16 for each PE (16*64 -> 9*64) for debug output
        /*
        std::vector<int> gdb;
        std::vector<int> tmp = ipa_.fm_buf_.AsVec(0, 1024, 8);
        for (int ii = 0; ii < 64; ii++) {
          for (int jj = 0; jj < 9; jj++) {
            gdb.push_back(tmp[ii * 16 + jj]);
          }
        }
        writeOut<int, 2>(os, gdb, dump);
        */
        
        // Get wgt addr
        wgt_addr = compute->ker_addr_s + p;
        // Split ker
        int8_t* wgt_src_a = reinterpret_cast<int8_t*>(
            ipa_dw_.wgt_buf_a_.BeginPtr(0));
        int8_t* wgt_src_b = reinterpret_cast<int8_t*>(
            ipa_dw_.wgt_buf_b_.BeginPtr(0));
        int8_t* wgt = reinterpret_cast<int8_t*>(wgt_ram_dw_.BeginPtr(wgt_addr));
        int jj = 0;
        for (int ii = 0; ii < 64 * ker_cnt; ii++) {
          wgt_src_a[jj] = wgt[2 * ii];
          wgt_src_b[jj] = wgt[2 * ii + 1];
          if ((jj + 1) % ker_cnt == 0) {
            for (int u = 0; u < 16 - ker_cnt; u++) {
              wgt_src_a[jj] = 0;
              wgt_src_b[jj] = 0;
              jj++;
            }  
          } else {
            jj++;
          }
        }
        /*
        std::vector<int> tmp = wgt_ram->AsVec(wgt_addr, 2048, 8);
        std::vector<int> gdb(tmp.begin(), tmp.begin() + 1152);
        writeOut<int, 2>(os, gdb, dump);
        */
        
        // Compute
        ipa_dw_.Forward(2 << compute->output_num);

        if (ipa_.GetOutputNum() >= compute->output_channel) {
          // Load out_adder_b
          //   - bias : loaded at the beginning
          //   - temp : partial sum from tmp_buf_
          ipa_dw_.adder_b.clear();
          if (compute->add_bias) {
            for (auto value : bias) {
              ipa_dw_.adder_b.push_back(
                  Saturate(value << compute->shift_num_bias, value > 0, 26));
            }
          } else if (compute->add_temp) {
            ipa_dw_.adder_buf_b_.Load<1024>(MemOp(0, 1, 0),
                tmp_buf_.BeginPtr(tmp_addr));
            std::vector<int> tmp = ipa_dw_.adder_buf_b_.AsVec(0, 64, 16);
            for (auto value : tmp) {
              ipa_dw_.adder_b.push_back(static_cast<int>(value) << 10);
            }
          }

          // Load adder_a from ipa output
          ipa_dw_.Accumulate(
                      compute->shift_num_fm,
                      compute->final_output,
                      tmp_buf_.BeginPtr(tmp_addr), os, dump);
          tmp_addr++;
        }
      }
    }
  }
  
  // cycle count
  int64_t ii = 1;
  int64_t epilogue = 30;
  int64_t latency = compute->dma_block_x_size*2 +2 + initiate_cnt * ii + epilogue;
  profiler.collect(latency);
}


/*
 * Top function wrapper
 */
void Device::Run(int parallel_block_cnt) {
  for (int u = 0; u < parallel_block_cnt; u++) {
    std::cout <<
        "*************************Parallel Block**********************\n";
    if (!event_q.empty()) {
      std::vector<OPUGenericInsn*> ins_vec = event_q.front();
      event_q.pop();
      // Arbiter
      for (auto ins : ins_vec) {
        if (ins->IsInstance<OPUComputeInsn>()) {
          OPUComputeInsn* compute = static_cast<OPUComputeInsn*>(ins);
          int fid = (fm_ram_id + fm_ram_vec_.size() - 1)
            % fm_ram_vec_.size();
          int wid = (wgt_ram_id + wgt_ram_vec_.size() - 1)
            % wgt_ram_vec_.size();
          int bid = (bias_ram_id + bias_ram_vec_.size() - 1)
            % bias_ram_vec_.size();
          // Consecutive dma for nxn conv, fetching from same fm buffer
          if (reg_.dma_start_trig == OPU_DMA_TRIG_DMA) {
            fid = compute->fm_ram_id;
            wid = compute->wgt_ram_id;
            bid = compute->bias_ram_id;
          }
          compute->SwitchUpdate(fid, wid, bid);
        }
      }
      // Run modules in parallel semantic (seq in real)
      for (auto ins : ins_vec) {
        if (ins->IsInstance<OPUDDRLDInsn>()) {
          OPUDDRLDInsn* load = static_cast<OPUDDRLDInsn*>(ins);
          RunLoad(load);
        } else if (ins->IsInstance<OPUDDRSTInsn>()) {
          OPUDDRSTInsn* store = static_cast<OPUDDRSTInsn*>(ins);
          RunStore(store);
        } else if (ins->IsInstance<OPUComputeInsn>()) {
          OPUComputeInsn* compute = static_cast<OPUComputeInsn*>(ins);
          RunCompute(compute);
        } else {
          std::cout << "[ERROR] unknown instruction type!";
          exit(1);
        }
      }
      FetchInsn();
      profiler.sync();
      // Propagate completion
      std::vector<OPUGenericInsn*> ins_vec_next = DependencyCheck(ins_vec);
      if (ins_vec_next.size() > 0) {
        event_q.push(ins_vec_next);
      }
    } else {
      compute_finish = false;
      compute_cnt = 0;
      compute_cnt_enable = false;
      break;
    }
  }
  profiler.dump();
}

/*
 * Propagate completion and get new events to run
 */
std::vector<OPUGenericInsn*>
    Device::DependencyCheck(std::vector<OPUGenericInsn*> ins_vec) {
  std::stringstream os;
  std::vector<OPUGenericInsn*> ins_vec_next;
  for (auto ins : ins_vec) {
    os << GetInsName(ins) << "\n";
    // Propagate completion of ins to its successors
    for (auto succ : ins->succ_) {
      os << "[check succ]" << GetInsName(succ) << "\n";
      succ->pred_[ins] = true;
      bool sat = false;
      if (succ->pred_.size() > 0) {
        sat = true;
        for (auto item : succ->pred_) {
          sat &= item.second;
          os << "\t[succ pred]" << GetInsName(item.first) << ":"
             << item.second << "\n";
        }
      }
      // Check pre-condition assertions
      if (sat) {
        os << "\tcheck " << succ->assert_.size() << " assertion(s)\n";
        if (succ->assert_.size() > 0) {
          os << "compute_cnt = " << compute_cnt << "\n";            
          for (auto item : succ->assert_) {
            if (static_cast<bool*>(item.first)) {
              if (static_cast<int>(*static_cast<bool*>(item.first))
                  != item.second) {
                os << "\t$" << static_cast<int>(*static_cast<bool*>(item.first))
                   << " != " << item.second << "\n";
                sat = false;
                break;
              }
            } else if (static_cast<int*>(item.first)) {
              if (*static_cast<int*>(item.first) != item.second) {
                os << "\t" << *static_cast<int*>(item.first) << " != "
                   << item.second << ">>" << compute_cnt << "\n";
                sat = false;
                break;
              } else {
                os << "\treset int\n";
                compute_cnt = 0;
                compute_cnt_enable = false;
              }
            } else {
              os << "\tELSE\n";
            }
          }
        }
      }
      // If all pre-conditions satisfied, schedule succ to next
      if (sat) {
        auto it = std::find(ins_vec_next.begin(),
                            ins_vec_next.end(),
                            succ);
        if (it == ins_vec_next.end()) {
          ins_vec_next.push_back(succ);
        }
      }
    }
  }
  if (compute_cnt == reg_.ddr_load_start_dma_num) {
    compute_cnt = 0;
  }
  // if (dump) std::cout << os.str();
  return ins_vec_next;
}

/*
 * Fetch one instruction block
 */
void Device::FetchInsn() {
  std::cout << "************BEGIN OF ONE INS BLK************\n";
  int ins_rd_cnt = 0;
  // Fetch until ins->immi == 0
  while (1) {
    std::cout << "ins_ram addr: " << ins_pc << "\n";
    ins_rd_cnt++;
    // Fetch one short instruction
    uint32_t data = *static_cast<uint32_t*>(ins_ram_.BeginPtr(ins_pc));
    OPUShortInsn* ins = new OPUShortInsn(data);
    // Decode and update register file
    ins->Decode(&reg_);
    // Update control flow
    DependencyUpdate(ins);
    ins_pc++;
    if (!ins->immi) {
      load_ins_->ReadFrom(&reg_);
      store_ins_->ReadFrom(&reg_);
      compute_ins_->ReadFrom(&reg_);
      DependencyUpdateUtil();
      if (layer_start_) {
        event_q.push({load_ins_});
        layer_start_ = false;
      }
      break;
    }
  }
  std::cout << "************END OF ONE INS BLK************\n";
  
  // cycle count
  profiler.collect(ins_rd_cnt);
}

/*
 * Check if the whole model is completed
 */
bool Device::IsComplete() {
  return reg_.network_done;
}

/*
 * Parse trigger conditions embedded in instructions
 */
void Device::DependencyUpdate(OPUShortInsn* ins) {
  if (ins->opcode == 11) {
    load_ins_->SetAssert({});
    if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_DDRLD) {
      load_ins_->SetPred({load_ins_});
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_DDRLD_DMA) {
      load_ins_->SetPred({load_ins_, compute_ins_});
      load_ins_->SetAssert(
        {{static_cast<void*>(&compute_cnt), reg_.ddr_load_start_dma_num }});
      compute_cnt_enable = true;
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_LAYER_START) {
      layer_start_ = true;
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_DDRST) {
      load_ins_->SetPred({store_ins_});
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_BRAMST) {
      load_ins_->SetPred({compute_ins_});
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_BRAMWB_DDRLD) {
      load_ins_->SetPred({load_ins_, store_ins_});
    } else if (reg_.ddr_load_start_trig == OPU_DDRLD_TRIG_DDRLD_DDRST) {
      load_ins_->SetPred({load_ins_, store_ins_});
    } else {
      load_ins_->SetPred({});
    }
  } else if (ins->opcode == 19) {
    store_ins_->SetAssert({});
    if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_BRAMST_DDRLD) {
      store_ins_->SetPred({compute_ins_, load_ins_});
      store_ins_->SetAssert({{static_cast<void*>(&compute_finish), 1}});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_DDRST) {
      store_ins_->SetPred({store_ins_});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_BRAMWB_DDRLD) {
      store_ins_->SetPred({load_ins_, store_ins_});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_DDRLD) {
      store_ins_->SetPred({load_ins_});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_DDRLD_DDRST) {
      store_ins_->SetPred({load_ins_, store_ins_});
    } else if (reg_.ddr_save_start_trig == OPU_DDRST_TRIG_SINGLE_DDRLD) {
      store_ins_->SetPred({load_ins_});
      store_ins_->SetAssert(
        {{static_cast<void*>(&load_single), 1}});
    } else {
      store_ins_->SetPred({});
    }
  } else if (ins->opcode == 12) {
    compute_ins_->SetAssert({});
    if (reg_.dma_start_trig == OPU_DMA_TRIG_DMA) {
      compute_ins_->SetPred({compute_ins_});
    } else if (reg_.dma_start_trig == OPU_DMA_TRIG_DDRLD) {
      compute_ins_->SetPred({load_ins_});
    } else if (reg_.dma_start_trig == OPU_DMA_TRIG_DMA_DDRLD) {
      compute_ins_->SetPred({compute_ins_});
      compute_ins_->SetAssert(
        {{static_cast<void*>(&compute_cnt), reg_.ddr_load_start_dma_num }});
      compute_cnt_enable = true;
    } else if (reg_.dma_start_trig == OPU_DMA_TRIG_DDRST_NOT_1_6) {
      compute_ins_->SetPred({store_ins_});
    } else if (reg_.dma_start_trig == OPU_DMA_TRIG_DDRST) {
      compute_ins_->SetPred({store_ins_});
    } else {
      compute_ins_->SetPred({});
    }
  }
}

/*
 * Update events to trigger
 */
void Device::DependencyUpdateUtil() {
  for (auto ins : ins_vec_) {
    ins->succ_.clear();
  }
  for (auto ins : ins_vec_) {
    for (auto item : ins->pred_) {
      item.first->succ_.push_back(ins);
    }
  }
}

/*
 * Get instruction name
 */
std::string Device::GetInsName(OPUGenericInsn* ins) {
  if (ins->IsInstance<OPUDDRLDInsn>()) {
    return "DDR_LD_INSN";
  } else if (ins->IsInstance<OPUDDRSTInsn>()) {
    return "DDR_ST_INSN";
  } else if (ins->IsInstance<OPUComputeInsn>()) {
    return "COMPUTE";
  } else {
    return "UNKNOWN";
  }
}
