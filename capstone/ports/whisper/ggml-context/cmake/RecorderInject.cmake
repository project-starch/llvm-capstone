# Deferred until upstream has created its ordinary ggml-base target.
function(wg_attach_recorder)
  target_sources(ggml-base PRIVATE "${WG_PORT_ROOT}/src/native/record.c")
  target_include_directories(ggml-base PRIVATE "${WG_PORT_ROOT}/src/native" "${WG_PORT_ROOT}/src/shared")
  target_compile_definitions(ggml-base PRIVATE WG_RECORD)
endfunction()
cmake_language(DEFER CALL wg_attach_recorder)
