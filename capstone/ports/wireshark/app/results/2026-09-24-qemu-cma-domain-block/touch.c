/* M-infra gate domain: a stack frame of FRAME bytes, touched at both ends. dom_data is the
   stack (my_first_domain/start.S takes sp from the monitor's dom_data), so if the monitor's
   split left dom_data smaller than the frame, big[0] lies below its base and faults. */
#ifndef FRAME
#error "-DFRAME=<bytes>"
#endif
void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile char big[FRAME];
  big[0] = 1;
  big[FRAME - 1] = 2;
  *res = 0xB16 + big[0] + big[FRAME - 1];   /* 0xB19 = 2841 when both touches landed */
}
