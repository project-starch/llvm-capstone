/* Compatibility include for existing musl probe and application loaders. */
#ifndef CAPSTONE_MUSL_HOST_SERVICE_H
#define CAPSTONE_MUSL_HOST_SERVICE_H
#include "../../../tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h"
#include "../../../runtime/linux/host-service.h"
#define HOSTCALL_FILE_SERVICE_PROBE_MAX_HANDLES HC_FILE_SERVICE_MAX_HANDLES
#endif
