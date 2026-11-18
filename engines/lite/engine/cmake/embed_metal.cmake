# Writes the Metal shader source SRC into DST as a C++ raw string literal, for metal_backend.mm to compile at run time.
# Usage: cmake -DSRC=kernels.metal -DDST=metal_kernels_src.inc -P embed_metal.cmake
file(READ "${SRC}" _metal_src)
string(FIND "${_metal_src}" ")KTMSL\"" _clash)
if(NOT _clash EQUAL -1)
    message(FATAL_ERROR "${SRC} contains the raw-string end marker )KTMSL\"")
endif()
file(WRITE "${DST}.tmp" "R\"KTMSL(${_metal_src})KTMSL\"\n")
configure_file("${DST}.tmp" "${DST}" COPYONLY)
file(REMOVE "${DST}.tmp")
