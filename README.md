# OAK-D Lite: Technical & Vision Context Report for Capstone

## 1. Core Hardware Specifications
The OAK-D Lite is an edge AI vision system driven by the Intel Movidius Myriad X VPU. It offloads all computer vision (CV) and neural network (NN) processing from your host MCU/SBC.



* **VPU:** Intel Movidius Myriad X (4 TOPS total; 1.4 TOPS for AI inference).
* **RGB Camera (Center):** 13 MP Sony IMX214.
    * Resolution: 4208x3120.
    * Video Encoding: 4K/30fps, 1080p/60fps (H.264, H.265, MJPEG).
    * Shutter: Rolling.
    * Focus: Available in Auto-Focus (AF: 8cm to infinity) or Fixed-Focus (FF: 50cm to infinity). Note: Use FF for high-vibration environments like rovers or drones.
* **Stereo Depth Cameras (Left/Right):** 2x OmniVision OV7251.
    * Resolution: 640x480 (480p).
    * Framerate: Up to 120 FPS.
    * Shutter: Global (crucial for high-speed tracking).
    * Baseline: 75 mm.
* **Interface & Power:** USB-C (USB 3.1 Gen 1). Draws 2.5W to 5W depending on VPU load. If using a Raspberry Pi as the host, utilize a Y-adapter to supply external power, as standard USB 2.0 ports cap at 900mA.

## 2. Vision Context & Architecture
For a capstone project, you do not interface with this like a standard UVC webcam. It uses the DepthAI API, which operates on a node-based pipeline architecture. You define a graph of operations and link them together.



### 2.1 Standard Vision Capabilities
* **Stereo Depth Perception:** Generates depth maps in real-time. Ideal range is 40cm to 8m. Depth error is under 2% below 3m, scaling to under 6% at 8m.
* **Spatial AI (3D Localization):** Combines 2D object detection (e.g., YOLO, MobileNet) with the depth map to provide physical 3D coordinates (X, Y, Z in meters) of the bounding box relative to the camera center.
* **Hardware Acceleration:** Warp, dewarp, resize, and crop (ImageManip node) execute directly on the VPU before feeding frames to the NN, saving host CPU cycles.
* **Object Tracking:** Zero-overhead 2D and 3D object tracking (ObjectTracker node) handles association across frames without host intervention.

### 2.2 Pipeline Implementation Logic
Here is the baseline logic for setting up a vision pipeline. You build the pipeline on the host, serialize it, and flash it to the OAK-D Lite via USB.

    import depthai as dai

    # 1. Initialize Pipeline
    pipeline = dai.Pipeline()

    # 2. Create Nodes
    camRgb = pipeline.create(dai.node.ColorCamera)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    # 3. Configure Nodes
    xoutDepth.setStreamName("depth")
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # 4. Link Nodes
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(xoutDepth.input)

    # 5. Connect and Execute
    with dai.Device(pipeline) as device:
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        while True:
            inDepth = qDepth.tryGet()
            if inDepth is not None:
                # Extract depth array for processing
                depthFrame = inDepth.getFrame()

## 3. Integration Strategy for Capstone
1.  **Model Selection:** The Myriad X requires models in the `.blob` format (OpenVINO IR). Train your model in PyTorch/TensorFlow, export to ONNX, and use the Luxonis Model Optimizer to convert it.
2.  **Host Machine Load:** A standard SBC (like a Raspberry Pi 4) is sufficient. The Pi only needs to handle the output queue (receiving the bounding boxes/coordinates or lightweight depth arrays) while the OAK-D Lite handles the heavy matrix multiplications.
3.  **Thermal Management:** The device can run hot under full VPU utilization (up to 105 degrees Celsius maximum operating temperature for the VPU chip itself). Design your capstone mount to allow passive airflow around the aluminum housing.
