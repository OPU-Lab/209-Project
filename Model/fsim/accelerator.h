#include <string>
#include <fstream>
#include <queue>
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <climits>

#include "./hw_spec.h"
#include "./instruction.h"
#include "./bit_util.h"
#include "./vmem.h"
#include "./config.h"
#include "./profiler.h"

#ifndef FSIM_ACCELERATOR_H_
#define FSIM_ACCELERATOR_H_
/*
 * Define template components for hardware modules, including SRAM, IPA
 */

/*
 * MemOp class encapsulates source/destination address and size for memory operation
 */ 
class MemOp {
 public:
  uint32_t src_addr;
  uint32_t size;
  uint32_t dst_addr;
  MemOp(uint32_t src_addr, uint32_t size, uint32_t dst_addr) {
    this->src_addr = src_addr;
    this->size = size;
    this->dst_addr = dst_addr;
  }
};

/*
 * reference: tsim for vta
 *   https://github.com/apache/incubator-tvm-vta/blob/master/src/sim/sim_driver.cc
 * 
 * SRAM class is parameterized with 
 *   - bits : bits per address for one bank
 *   - size : total address number
 *   - banks : memory banks number
 *   (assume data at same address from different banks are concatenated)
 */
template<int bits, int size, int banks>
class SRAM {
 public:
  static const int dBytes = bits * banks / 8;
  using DType = typename std::aligned_storage<dBytes, dBytes>::type;

  SRAM() {
    data_ = new DType[size];
  }

  ~SRAM() {
    delete [] data_;
  }

  // Get the i-th index
  void* BeginPtr(uint32_t index) {
    // CHECK_LT(index, kMaxNumElem);
    return &(data_[index]);
  }

  // Copy data from disk file to SRAM
  void InitFromFile(std::string filename) {
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    // read the data:
    file.read(reinterpret_cast<char*>(data_), fileSize);
    file.close();
  }

  // Copy data from disk file to SRAM with address offset and size specified
  //   - src_wl : word length corresponding to mem.src and mem.size
  template<uint32_t src_wl>
  void LoadFromFile(MemOp mem, const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    file.seekg(mem.src_addr * src_wl / 8, std::ios::beg);
    int bytes = mem.size * src_wl / 8;
    file.read(reinterpret_cast<char*>(data_) +
        mem.dst_addr * dBytes, bytes);
    file.close();
  }

  // Copy data from in-memory pointer to SRAM
  template<int src_bits>
  void Load(MemOp mem, void* src) {
    int8_t* src_t = reinterpret_cast<int8_t*>(src);
    size_t bytes = mem.size * src_bits / 8;
    std::memcpy(data_ + mem.dst_addr, src_t, bytes);
  }

  // Get data as std::vector<int> to compute
  std::vector<int> AsVec(int addr, int vec_size, int src_wl) {
    int* data = new int[vec_size];
    DType* src = data_ + addr;
    for (int i = 0; i < vec_size; i++) {
      if (src_wl == 8) {
        data[i] = static_cast<int>((reinterpret_cast<int8_t*>(src))[i]);
      } else if (src_wl == 16) {
        int16_t value = (reinterpret_cast<int16_t*>(src))[i];
        value = ((value & 0xFF) << 8) | ((value & 0xFF00) >> 8);
        data[i] = static_cast<int>(value);
      }
    }
    std::vector<int> vec(vec_size);
    std::memcpy(&vec[0], data, sizeof(int)*vec_size);
    delete [] data;
    return vec;
  }

 private:
  DType* data_;
};

/*
 * IPA class is parameterized with
 *  - mac : number of multiply-accumulate units (DSP macros on FPGA) per PE
 *  - pe : number of PE (processing unit), which is a bunch of macs followed 
 *         by adder tree for reduction
 *  - operand_bits : word lengths for mac operands (8 by default)
 */
template<int mac, int pe, int operand_bits>
class IPA {
 public:
  static const int nMac = mac * pe;
  // Compute buffer
  using PE_buf_t = SRAM<mac*operand_bits, 1, pe>;
  PE_buf_t fm_buf_;
  PE_buf_t wgt_buf_a_;
  PE_buf_t wgt_buf_b_;
  using Acc_buf_t = SRAM<32*64, 1, 1>;
  Acc_buf_t adder_buf_a_;
  Acc_buf_t adder_buf_b_;

  std::vector<int> psum;
  std::vector<int> adder_b;

  // Get current valid ipa output number
  int GetOutputNum() {
    return psum.size();
  }

  // Fetch data from compute buffer and get ipa results
  void Forward(int output_num) {
    std::vector<int> ifm = fm_buf_.AsVec(0, nMac, 8);
    std::vector<int> wgt_a = wgt_buf_a_.AsVec(0, nMac, 8);
    std::vector<int> wgt_b = wgt_buf_b_.AsVec(0, nMac, 8);

    std::vector<int> psum_local;
    size_t step = nMac / (output_num / 2);
    for (int i = 0; i < output_num / 2; i++) {
      int sum = std::inner_product(ifm.begin() + i * step,
              ifm.begin() + (i + 1) * step, wgt_a.begin() + i * step, 0);
      psum_local.push_back(sum);
      sum = std::inner_product(ifm.begin() + i * step,
              ifm.begin() + (i + 1) * step, wgt_b.begin() + i * step, 0);
      psum_local.push_back(sum);
    }
    
    // Concatenate at front
    psum.insert(psum.begin(), psum_local.begin(), psum_local.end());
  }

  // Output control for accumulation
  void Accumulate(int fm_lshift_num, bool cut, void* dst,
    std::ofstream& os, bool dump) {
    std::vector<int> adder_a(64 - psum.size(), 0);
    for (int i = 0; i < psum.size(); i++) {
      int value = psum[i];
      adder_a.push_back(
        Saturate(value << fm_lshift_num, value > 0, 26));
    }
    // std::vector<int> gdb = wgt_ram.AsVec(wgt_addr, 1024, 8);
#ifdef DEBUG_OUT_ADDER_B
    writeOut<int, 8>(os, adder_b, dump);
#endif
#ifdef DEBUG_OUT_ADDER_A
    writeOut<int, 8>(os, adder_a, dump);
#endif
    psum.clear();
    std::vector<int> res;
    auto ia = adder_a.begin();
    auto ib = adder_b.begin();
    for (int ii = 0; ii < 64; ii++) {
      int p = *(ia++) + *(ib++);
      res.push_back(p);
    }
    // 27b--cut->16b
    for (auto &item : res) {
      item = Saturate(item >> 10, item > 0, 16);
    }
#ifdef DEBUG_CUT_16
    writeOut<int, 4>(os, res, dump);
#endif
    if (cut) {
      // 16b--round->8b
      for (auto &item : res) {
        bool positive = item > 0;
        if (item & 0x80)
          item = (item >> 8) + 1;
        else
          item = item >> 8;
        item = Saturate(item, positive, 8);
      }
    }
#ifdef DEBUG_CUT_8
    writeOut<int, 2>(os, res, dump);
#endif
    // write to dst* of temp buffer
    std::vector<int16_t> trunc;
    for (auto item : res) {
      // -> little endian
      int16_t value = static_cast<int16_t>(item);
      value = ((value & 0xFF) << 8) | ((value & 0xFF00) >> 8);
      trunc.push_back(value);
    }
    std::memcpy(dst, &trunc[0], 64*16/8);  // byte count
  }
};

/*
 * Device class is the top abstraction for OPU overlay
 */
class Device {
 public:
  uint32_t ins_pc{0};
  // Special purpose registers
  OPURegFile reg_;
  // Scratchpad memory
  using Ins_ram_t = SRAM<32, 2<<15, 1>;
  using Fm_ram_t = SRAM<512, 4096, 1>;
  using Wgt_ram_t = SRAM<512, 64, 16>;
  using Bias_ram_t = SRAM<1024, 1, 1>;
  Ins_ram_t ins_ram_;
  Fm_ram_t fm_ram_a_;
  Fm_ram_t fm_ram_b_;
  Wgt_ram_t  wgt_ram_a_;
  Wgt_ram_t  wgt_ram_b_;
  Bias_ram_t bias_ram_a_;
  Bias_ram_t bias_ram_b_;
  // IPA
  using IPA_t = IPA<16, 32, 8>;
  IPA_t ipa_;
  // Partial sum buffer
  using Tmp_buf_t = SRAM<1024, 2<<11, 1>;
  Tmp_buf_t tmp_buf_;
  // DDR
  using DRAM = VirtualMemory;
  DRAM dram;
  
  // dw
  using Wgt_ram_dw_t = SRAM<1024, 64, 16>;
  Wgt_ram_dw_t wgt_ram_dw_;
  using IPA_DW_t = IPA<16, 64, 8>;
  IPA_DW_t ipa_dw_;

  // Double buffering
  std::vector<Fm_ram_t*> fm_ram_vec_;
  std::vector<Wgt_ram_t*> wgt_ram_vec_;
  std::vector<Bias_ram_t*> bias_ram_vec_;
  std::vector<Ins_ram_t*> ins_ram_vec_;
  size_t fm_ram_id{0};
  size_t wgt_ram_id{0};
  size_t bias_ram_id{0};
  size_t ins_ram_id{0};

  // Control flow
  std::queue<std::vector<OPUGenericInsn*>> event_q;
  OPUDDRLDInsn* load_ins_;
  OPUDDRSTInsn* store_ins_;
  OPUComputeInsn* compute_ins_;
  std::vector<OPUGenericInsn*> ins_vec_;
  void DependencyUpdate(OPUShortInsn* ins);
  void DependencyUpdateUtil();
  std::vector<OPUGenericInsn*>
    DependencyCheck(std::vector<OPUGenericInsn*> ins_vec);
  // Global variables
  bool layer_start_{false};
  bool compute_finish {false};
  bool compute_cnt_enable {false};
  int compute_cnt {0};
  bool load_single {false};
  int fc_tmp_addr {0};

  // Function wrappers
  void Run(int cnt = INT_MAX);
  void FetchInsn();
  void RunInsn(OPUGenericInsn* ins);
  void RunLoad(OPUDDRLDInsn* load);
  void RunStore(OPUDDRSTInsn* store);
  void RunPostProcess(OPUDDRSTInsn* store);
  void RunPadding(OPUDDRSTInsn* store);
  void RunCompute(OPUComputeInsn* compute);
  void RunComputeDW(OPUComputeInsn* compute);
  void RunComputeFC(OPUComputeInsn* compute);
  // Sub-function wrappers
  void Pooling(std::vector<int>& data_o, std::vector<int> data_i, int type,
    bool st);
  void Activation(std::vector<int>& data, int type);
  void ResidualAdd(std::vector<int>& data, bool enable, int addr);

  // Utility function
  std::string GetInsName(OPUGenericInsn* ins);
  bool IsComplete();

  // Debug
  std::ofstream os;
  bool dump {false};
  bool skip_exec {false};
  Profiler profiler;

  Device() {
    dram.Init();
    // srams
    //ins_ram_.InitFromFile("tinyyolo/ins.bin");
    //ins_ram_.InitFromFile("yolov3/ins.bin");
    ins_ram_.InitFromFile(INS_FILE_PATH);
    ins_ram_vec_ = {&ins_ram_};
    fm_ram_vec_ = {&fm_ram_a_, &fm_ram_b_};
    wgt_ram_vec_ = {&wgt_ram_a_, &wgt_ram_b_};
    bias_ram_vec_ = {&bias_ram_a_, &bias_ram_b_};
    // coarse-grained generic instructions
    load_ins_ = new OPUDDRLDInsn();
    store_ins_ = new OPUDDRSTInsn();
    compute_ins_ = new OPUComputeInsn();
    ins_vec_ = {load_ins_, store_ins_, compute_ins_};
    
    os.open("out.txt");
  }
};
#endif  // FSIM_ACCELERATOR_H_
