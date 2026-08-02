#include <torch/library.h>

#include <string>

namespace worldfoundry_native {

std::string build_info();

}  // namespace worldfoundry_native

TORCH_LIBRARY_FRAGMENT(worldfoundry_native, m) {
  m.def("_build_info() -> str", &worldfoundry_native::build_info);
}

