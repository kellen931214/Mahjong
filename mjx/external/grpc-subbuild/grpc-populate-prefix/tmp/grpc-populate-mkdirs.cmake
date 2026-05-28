# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/workspace/Mahjong/mjx/external/grpc-src")
  file(MAKE_DIRECTORY "/workspace/Mahjong/mjx/external/grpc-src")
endif()
file(MAKE_DIRECTORY
  "/workspace/Mahjong/mjx/external/grpc-build"
  "/workspace/Mahjong/mjx/external/grpc-subbuild/grpc-populate-prefix"
  "/workspace/Mahjong/mjx/external/grpc-subbuild/grpc-populate-prefix/tmp"
  "/workspace/Mahjong/mjx/external/grpc-subbuild/grpc-populate-prefix/src/grpc-populate-stamp"
  "/workspace/Mahjong/mjx/external/grpc-subbuild/grpc-populate-prefix/src"
  "/workspace/Mahjong/mjx/external/grpc-subbuild/grpc-populate-prefix/src/grpc-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/Mahjong/mjx/external/grpc-subbuild/grpc-populate-prefix/src/grpc-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/Mahjong/mjx/external/grpc-subbuild/grpc-populate-prefix/src/grpc-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
