#include <string>

#ifndef WORLDFOUNDRY_NATIVE_BUILD_ID
#define WORLDFOUNDRY_NATIVE_BUILD_ID "unknown"
#endif

#ifndef WORLDFOUNDRY_NATIVE_OPERATOR_ABI_VERSION
#define WORLDFOUNDRY_NATIVE_OPERATOR_ABI_VERSION 0
#endif

#define WORLDFOUNDRY_STRINGIFY_IMPL(value) #value
#define WORLDFOUNDRY_STRINGIFY(value) WORLDFOUNDRY_STRINGIFY_IMPL(value)

namespace worldfoundry_native {

std::string build_info() {
  return std::string{"{\"build_id\":\""} + WORLDFOUNDRY_NATIVE_BUILD_ID +
      "\",\"operator_abi_version\":" + WORLDFOUNDRY_STRINGIFY(WORLDFOUNDRY_NATIVE_OPERATOR_ABI_VERSION) + "}";
}

}  // namespace worldfoundry_native

