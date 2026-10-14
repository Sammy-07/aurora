#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

namespace {

bool compile_kernel(id<MTLDevice> device, const char* label, const char* source) {
  std::printf("\n--- %s ---\n", label);

  NSError* error = nil;
  NSString* sourceString = [NSString stringWithUTF8String:source];
  id<MTLLibrary> library = [device newLibraryWithSource:sourceString options:nil error:&error];
  if (library == nil) {
    std::printf("MSL library compilation failed: %s\n", error.localizedDescription.UTF8String);
    return false;
  }

  id<MTLFunction> function = [library newFunctionWithName:@"probe"];
  if (function == nil) {
    std::puts("MSL library compiled, but the probe function was not found.");
    return false;
  }

  error = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (pipeline == nil) {
    std::printf("Compute pipeline creation failed: %s\n", error.localizedDescription.UTF8String);
    return false;
  }

  std::printf("Compute pipeline creation succeeded (max threads: %lu).\n",
              static_cast<unsigned long>(pipeline.maxTotalThreadsPerThreadgroup));
  return true;
}

}  // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::puts("No default Metal device is available.");
      return 2;
    }

    std::printf("Metal device: %s\n", device.name.UTF8String);
    constexpr char kernelWithoutMetadata[] = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void probe() {}
)metal";
    constexpr char kernelWithMetadata[] = R"metal(
#include <metal_stdlib>
using namespace metal;
[[max_total_threads_per_threadgroup(1)]]
kernel void probe() {}
)metal";

    const bool withoutMetadata =
        compile_kernel(device, "Without max_total_threads_per_threadgroup", kernelWithoutMetadata);
    const bool withMetadata =
        compile_kernel(device, "With max_total_threads_per_threadgroup(1)", kernelWithMetadata);

    std::printf("\nResult: without metadata %s; with metadata %s.\n",
                withoutMetadata ? "passed" : "failed", withMetadata ? "passed" : "failed");
    return withoutMetadata && !withMetadata ? 0 : 1;
  }
}
