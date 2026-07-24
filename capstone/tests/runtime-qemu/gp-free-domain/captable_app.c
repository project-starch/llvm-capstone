/* Stub for the capstone-c-style cap-table gp probe. The real workload lives in
   start-gpfree-captable.S (inline); domain_main is unused. __pad forces the image
   past 0x1000 so the (unchanged) monitor's fixed base+0x1000 SPLIT is valid. */
volatile const unsigned __pad[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}; // 64B .rodata
void domain_main(unsigned *res, unsigned func) { (void)res; (void)func; }
