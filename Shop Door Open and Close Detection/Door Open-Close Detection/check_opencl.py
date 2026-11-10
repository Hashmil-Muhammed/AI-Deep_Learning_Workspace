import cv2

print("OpenCV Version:", cv2.__version__)
print("OpenCL Available:", cv2.ocl.haveOpenCL())

cv2.ocl.setUseOpenCL(True)

print("OpenCL Enabled:", cv2.ocl.useOpenCL())

if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
    device = cv2.ocl.Device_getDefault()

    print("Device Name:", device.name())
    print("Device Vendor ID:", device.vendorID())
    print("Device Version:", device.version())
    print("Device Type:", device.type())
else:
    print("OpenCL is not available.")