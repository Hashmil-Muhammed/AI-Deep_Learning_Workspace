import pyopencl as cl

platforms = cl.get_platforms()

for platform in platforms:
    print("\nPlatform:", platform.name)

    for device in platform.get_devices():
        print("  Device:", device.name)
        print("  Vendor:", device.vendor)
        print("  Type:", cl.device_type.to_string(device.type))