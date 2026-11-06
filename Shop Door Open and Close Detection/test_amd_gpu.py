import pyopencl as cl
import numpy as np
import time

# Find AMD OpenCL platform
platforms = cl.get_platforms()

amd_platform = None

for platform in platforms:
    if "AMD" in platform.name.upper():
        amd_platform = platform
        break

if amd_platform is None:
    raise RuntimeError("AMD OpenCL platform not found!")

# Select AMD GPU
devices = amd_platform.get_devices(
    device_type=cl.device_type.GPU
)

if not devices:
    raise RuntimeError("No AMD GPU found!")

device = devices[0]

print("Selected GPU:", device.name)
print("Vendor:", device.vendor)
print("Version:", device.version)

# Create AMD GPU context and queue
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Test GPU calculation
size = 1_000_000

a = np.random.rand(size).astype(np.float32)
b = np.random.rand(size).astype(np.float32)
result = np.empty_like(a)

buffer_a = cl.Buffer(
    context,
    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
    hostbuf=a
)

buffer_b = cl.Buffer(
    context,
    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
    hostbuf=b
)

buffer_result = cl.Buffer(
    context,
    cl.mem_flags.WRITE_ONLY,
    result.nbytes
)

program = cl.Program(context, """
__kernel void add_arrays(
    __global const float *a,
    __global const float *b,
    __global float *result
)
{
    int i = get_global_id(0);
    result[i] = a[i] + b[i];
}
""").build()

start = time.perf_counter()

program.add_arrays(
    queue,
    (size,),
    None,
    buffer_a,
    buffer_b,
    buffer_result
)

cl.enqueue_copy(queue, result, buffer_result)
queue.finish()

elapsed = time.perf_counter() - start

print("GPU calculation successful!")
print("First 5 results:", result[:5])
print("Execution time:", round(elapsed * 1000, 2), "ms")

# Verify result
print("Result correct:", np.allclose(result, a + b))