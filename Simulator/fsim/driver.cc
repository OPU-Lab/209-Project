#include "accelerator.h"
/*
 * Top driver function for OPU simulation
 *
 * for each layer:
 *   dev->FetchInsn()  // fetch one instruction block
 *   dev->Run(#phases)  // run until no new events pop out by default
 *
 * dev->Run(N) runs N phases, which could include events in parallel
 * semantic. Events include IF (instruction fetch), LD (DDR load), ST
 * (DDR store), COMPUTE (pipeline : data fetch - compute - accumulate).
 * 
 *
 * for example, say the flow consists of 4 phases as follows
 * 
 * IF -- LD/IF -- LD/COMPUTE/IF -- COMPUTE/ST/IF
 *
 * dev->Run(4) simulates the flow above
 */
int main() {
  Device* dev = new Device();
  
  for (int i = 0; ; i++) {
    if (i == LAYER_CNT - 1) dev->dump = true;
    if (dev->IsComplete())
      break;
    dev->FetchInsn();
    dev->Run();
  }
  dev->os.close();
  return 0;
}
