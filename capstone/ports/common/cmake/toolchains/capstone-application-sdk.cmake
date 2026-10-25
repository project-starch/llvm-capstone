# Registers the application SDK as a build input. Reached through
# CMAKE_PROJECT_INCLUDE, which capstone-application.cmake sets and which CMake
# includes at the end of every project() call -- the earliest point where
# directory properties exist and the latest that still precedes any target.
# The reasoning for all three registrations is in capstone-application.cmake,
# beside the digest they use.
#
# A port that needs a project include of its own has to include this file from
# it: CMAKE_PROJECT_INCLUDE only became a list in CMake 3.29, and the ports
# build with 3.25.

# Reconfigure when any SDK file changes. Without this the digest below would
# keep whatever value it had at the first configure, and nothing else here
# would ever rerun.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CAPSTONE_SDK_ARTIFACTS})

# Recompilation: an SDK change changes every compile command in the directory.
# A define rather than a stamp file, so it reaches assembler sources too and
# needs no include path.
add_compile_definitions(CAPSTONE_SDK_IDENTITY=0x${CAPSTONE_SDK_IDENTITY}ULL)

# The link. Deferred to the end of the directory, by which point this
# directory's targets and all of its subdirectories' targets exist. LINK_DEPENDS
# is an order-only dependency: it retriggers the link without putting the
# archives on the command line, where they would be a second copy of what
# capstone-cc already links.
function(capstone_sdk_link_depends directory)
  get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_property(type TARGET ${target} PROPERTY TYPE)
    if(type STREQUAL "EXECUTABLE")
      set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${CAPSTONE_SDK_ARTIFACTS})
    endif()
  endforeach()
  get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(subdirectory IN LISTS subdirectories)
    capstone_sdk_link_depends("${subdirectory}")
  endforeach()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
               CALL capstone_sdk_link_depends "${CMAKE_CURRENT_SOURCE_DIR}")
