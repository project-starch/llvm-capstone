include_guard(GLOBAL)

option(CAPSTONE_DOMAIN_FAULT_RECOVERY
  "Enable cooperative domain fault returns (requires trap-delivery QEMU)" OFF)

# Configure an executable using the shared CALL/REGION_SHARE entry ABI.
# DATA_BYTES includes application globals/cap tables/stack, but not recovery
# state. STACK_BYTES is the diagnostic stack component of that requirement.
# Do not also add start.S, domreq.S, or define CAPSTONE_DOMREQ_* yourself.
function(capstone_configure_domain target)
  cmake_parse_arguments(PARSE_ARGV 1 domain "" "DATA_BYTES;STACK_BYTES" "")
  if(domain_UNPARSED_ARGUMENTS OR domain_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "Invalid capstone_configure_domain arguments")
  endif()
  foreach(size DATA_BYTES STACK_BYTES)
    if(NOT "${domain_${size}}" MATCHES "^[0-9]+$")
      message(FATAL_ERROR "capstone_configure_domain requires numeric ${size}")
    endif()
  endforeach()
  if(domain_STACK_BYTES LESS 96 OR domain_DATA_BYTES LESS domain_STACK_BYTES)
    message(FATAL_ERROR "Domain data must cover its stack, and stack must cover the entry frame")
  endif()
  get_target_property(configured ${target} CAPSTONE_DOMAIN_CONFIGURED)
  if(configured)
    message(FATAL_ERROR "Domain ${target} is already configured")
  endif()
  set_target_properties(${target} PROPERTIES CAPSTONE_DOMAIN_CONFIGURED TRUE)
  get_filename_component(capstone "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../.." ABSOLUTE)
  set(data_bytes ${domain_DATA_BYTES})
  if(CAPSTONE_DOMAIN_FAULT_RECOVERY)
    # Must match the reservation in the shared entry assembly.
    math(EXPR data_bytes "${data_bytes} + 256")
    target_compile_definitions(${target} PRIVATE CAPSTONE_DOMAIN_FAULT_RECOVERY)
  endif()
  target_sources(${target} PRIVATE
    "${capstone}/my_first_domain/start.S"
    "${capstone}/runtime/domain/gct-section-end.S"
    "${capstone}/runtime/domain/domreq.S")
  target_compile_definitions(${target} PRIVATE
    CAPSTONE_DOMREQ_DATA=${data_bytes} CAPSTONE_DOMREQ_STACK=${domain_STACK_BYTES})
  target_link_libraries(${target} PRIVATE Capstone::Runtime)
  set(link_script "${capstone}/my_first_domain/link.ld")
  target_link_options(${target} PRIVATE --gc-sections -T "${link_script}")
  set_target_properties(${target} PROPERTIES SUFFIX .dom LINK_DEPENDS "${link_script}")
endfunction()
