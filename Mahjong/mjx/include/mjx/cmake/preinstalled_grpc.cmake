set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)

# Find gRPC manually (system gRPC 1.30 doesn't provide CMake config files)
find_path(GRPC_INCLUDE_DIR grpc++/grpc++.h)
find_library(GRPC_LIBRARY grpc++ /usr/lib/x86_64-linux-gnu)
find_library(GRPC_UNSECURE_LIBRARY grpc++_unsecure /usr/lib/x86_64-linux-gnu)
find_library(GRPC_REFLECTION_LIBRARY grpc++_reflection /usr/lib/x86_64-linux-gnu)
find_library(GRPC_CORE_LIBRARY grpc /usr/lib/x86_64-linux-gnu)

if(NOT GRPC_LIBRARY)
  message(FATAL_ERROR "gRPC library not found")
endif()

message(STATUS "Using Protobuf ${Protobuf_VERSION}")
message(STATUS "gRPC include dir: ${GRPC_INCLUDE_DIR}")
message(STATUS "gRPC library: ${GRPC_LIBRARY}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
find_program(_PROTOBUF_PROTOC protoc)
set(_GRPC_GRPCPP ${GRPC_LIBRARY} ${GRPC_UNSECURE_LIBRARY} ${GRPC_CORE_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} protobuf::libprotobuf)
set(_REFLECTION ${GRPC_REFLECTION_LIBRARY})
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)