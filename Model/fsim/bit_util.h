#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#ifndef FSIM_BIT_UTIL_H_
#define FSIM_BIT_UTIL_H_

inline int Saturate(int x, bool positive, int wl) {
  int mask = 0xFFFFFFFF << (wl-1);
  if (x == 0) {
    return x;
  } else if (!positive && ((mask & x) != mask)) {
    return mask;  // negative max, keep negative when extended to 32 bit int
  } else if (positive && (mask & x)) {
    return ~mask;
  }
  return x;
}

template <typename T, int W>
  void writeOut(std::ofstream& os,
    std::vector<T> data, bool dump = false) {  // W - #char to print
    if (!dump) return;
    std::stringstream ss;
    for (auto item : data) {
      std::stringstream sstream;
      sstream << std::hex << static_cast<int>(item);
      std::string ret = sstream.str();
      if (ret.length() > W) {
        ret = ret.substr(ret.length() - W, W);
      } else if (ret.length() < W) {
        for (int ii = 0; ii <= W - ret.length(); ii++)
          ret = "0"+ret;
      }
      ss << ret << " ";
    }
    os << ss.str() << "\n";
  }
#endif  // FSIM_BIT_UTIL_H_
