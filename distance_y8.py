"""
distance_y8.py  -  Custom YOLOv8n "cube" detector with spatial distance (OAK-D)

Uses a custom-trained YOLOv8n blob (best_openvino_2022.1_6shave.blob) instead
of the model-zoo YOLOv6.  Because the blob is loaded raw via NeuralNetwork,
YOLOv8 output parsing (anchor decode + NMS) and spatial-coordinate calculation
(from the aligned depth map) are done on the host.

Key differences from distance.py:
  - NeuralNetwork node  (not SpatialDetectionNetwork)
  - Host-side YOLOv8 output decode + NMS
  - Depth-map ROI lookup for spatial XYZ
  - Single class: "cube"
"""

import depthai as dai
import cv2
import time
import math
import numpy as np
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
BLOB_PATH   = Path(__file__).parent / "best_openvino_2022.1_6shave.blob"
NN_SIZE     = 640       # must match the blob's input resolution
NUM_CLASSES = 1
CONF_THRESH = 0.1
IOU_THRESH  = 0.5   

labelMap = ["cube"]

# ── Coordinate Transform ────────────────────────────────────────────────────
# Camera frame (DepthAI spatial detection, RDF convention):
#   X_c = Right (+), Left (-)
#   Y_c = Down (+),  Up (-)
#   Z_c = Depth/Forward (+)
#
# With camera tilted down by pitch angle theta from horizontal, convert to
# 3D Cartesian (origin at camera, axes aligned to gravity):
#   X_3d = X_c                               (lateral, unchanged)
#   Y_3d = Z_c * sin(theta) + Y_c * cos(theta)  (vertical drop, + = down)
#   Z_3d = Z_c * cos(theta) - Y_c * sin(theta)  (horizontal forward)
# ─────────────────────────────────────────────────────────────────────────────

def accel_to_pitch(ax, ay, az):
    """Pitch (tilt-down angle) from raw accelerometer, in radians."""
    return math.atan2(az, math.sqrt(ax * ax + ay * ay))


def cam_to_3d(x_c, y_c, z_c, pitch_rad):
    """Camera-frame coords -> gravity-aligned 3D (mm)."""
    sp = math.sin(pitch_rad)
    cp = math.cos(pitch_rad)
    return x_c, z_c * sp + y_c * cp, z_c * cp - y_c * sp


# ── YOLOv8 Post-Processing ──────────────────────────────────────────────────

def parse_yolov8(raw_layer, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    """Decode raw YOLOv8 output into a list of detections.

    YOLOv8 output tensor: [4 + num_classes, num_anchors]  (8400 anchors for 640x640)
    Row layout: cx, cy, w, h, cls0_score, cls1_score, ...
    Box coordinates are in *pixel* space of the NN input (0..NN_SIZE).

    Returns
    -------
    list[dict]  with keys  x1, y1, x2, y2  (pixels in NN-input space),
                           confidence, label (int)
    """
    num_params = 4 + NUM_CLASSES          # 5 for single-class
    data = np.array(raw_layer, dtype=np.float32)
    num_anchors = len(data) // num_params
    output = data.reshape(num_params, num_anchors)   # [5, 8400]
    output = output.T                                 # [8400, 5]

    boxes  = output[:, :4]    # cx, cy, w, h
    scores = output[:, 4:]    # class scores

    class_ids   = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences > conf_thresh
    boxes       = boxes[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes) == 0:
        return []

    # cx,cy,w,h  ->  x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # OpenCV NMS expects [x, y, w, h]
    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(),
                               conf_thresh, iou_thresh)

    detections = []
    for idx in indices:
        i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
        detections.append({
            "x1": float(x1[i]), "y1": float(y1[i]),
            "x2": float(x2[i]), "y2": float(y2[i]),
            "confidence": float(confidences[i]),
            "label": int(class_ids[i]),
        })
    return detections


def roi_median_depth(depth_map, x1, y1, x2, y2,
                     min_mm=100, max_mm=15_000):
    """Median depth (mm) inside a ROI, ignoring invalid values."""
    h, w = depth_map.shape[:2]
    rx1, ry1 = max(0, int(x1)), max(0, int(y1))
    rx2, ry2 = min(w, int(x2)), min(h, int(y2))
    if rx1 >= rx2 or ry1 >= ry2:
        return 0.0
    roi = depth_map[ry1:ry2, rx1:rx2]
    valid = roi[(roi > min_mm) & (roi < max_mm)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


# ── Pipeline ─────────────────────────────────────────────────────────────────

with dai.Pipeline() as pipeline:

    # 1. Camera Nodes (v3 syntax)
    cam       = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoLeft  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # 2. Stereo Depth (no RGB alignment — stays in mono frame at 640x400)
    stereo = pipeline.create(dai.node.StereoDepth)
    monoLeft.requestOutput((640, 400)).link(stereo.left)
    monoRight.requestOutput((640, 400)).link(stereo.right)

    # 3. Neural Network  — custom YOLOv8n blob
    #    Camera outputs NV12 (1.5 B/px) but the blob expects 3-channel
    #    planar input (3 B/px).  ImageManip converts the colour format.
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(BLOB_PATH))

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(NN_SIZE * NN_SIZE * 3)  # 1,228,800 B
    cam.requestOutput((NN_SIZE, NN_SIZE)).link(manip.inputImage)
    manip.out.link(nn.input)

    # 4. IMU  — BMI270 raw accelerometer for pitch estimation
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # 5. Output Queues  (display at 1920x1080 to match original distance.py)
    DISP_W, DISP_H = 1920, 1080
    q_vid   = cam.requestOutput((DISP_W, DISP_H)).createOutputQueue(maxSize=4, blocking=False)
    q_nn    = nn.out.createOutputQueue(maxSize=4, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
    q_imu   = imu.out.createOutputQueue(maxSize=50, blocking=False)

    pipeline.start()

    # ── Camera intrinsics (for back-projecting to 3D) ────────────────────
    # Try device calibration first; fall back to estimated HFOV (~69 deg).
    try:
        calib = pipeline.getDefaultDevice().readCalibration()
        K = np.array(calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, NN_SIZE, NN_SIZE))
        fx, fy = K[0][0], K[1][1]
        cx_cam, cy_cam = K[0][2], K[1][2]
    except Exception:
        hfov_rad = math.radians(69)
        fx = fy = (NN_SIZE / 2) / math.tan(hfov_rad / 2)
        cx_cam = cy_cam = NN_SIZE / 2.0

    # Live pitch from IMU (smoothed with exponential moving average)
    pitch_rad = 0.0
    pitch_deg = 0.0
    ALPHA = 0.1

    depth_frame = None

    # 6. Processing Loop
    while pipeline.isRunning():
        # ── Video frame (1920x1080 display) ─────────────────────────
        vid_msg = q_vid.tryGet()
        if vid_msg is None:
            continue
        frame = vid_msg.getCvFrame()
        raw_frame = frame.copy()

        # ── Update pitch from raw accelerometer ──────────────────────
        imu_data = q_imu.tryGet()
        if imu_data is not None:
            for packet in imu_data.packets:
                acc = packet.acceleroMeter
                raw_pitch = accel_to_pitch(acc.x, acc.y, acc.z)
                pitch_rad = ALPHA * raw_pitch + (1 - ALPHA) * pitch_rad
                pitch_deg = math.degrees(pitch_rad)

        # ── Depth frame (aligned to RGB) ─────────────────────────────
        depth_msg = q_depth.tryGet()
        if depth_msg is not None:
            depth_frame = depth_msg.getCvFrame()

        # ── NN detections ────────────────────────────────────────────
        nn_msg = q_nn.tryGet()
        if nn_msg is not None:
            # DepthAI v3 NNData API — try common methods
            try:
                raw_layer = list(nn_msg.getTensor("output0").flatten())
            except Exception:
                try:
                    names = nn_msg.getAllLayerNames()
                    raw_layer = list(nn_msg.getTensor(names[0]).flatten())
                except Exception:
                    # Last resort: print available methods so we can fix
                    print("NNData methods:", [m for m in dir(nn_msg)
                          if "get" in m.lower() or "layer" in m.lower()
                          or "tensor" in m.lower()])
                    continue
            detections = parse_yolov8(raw_layer)

            for det in detections:
                # Normalize NN coords (0..640) then scale to display frame
                nx1 = det["x1"] / NN_SIZE
                ny1 = det["y1"] / NN_SIZE
                nx2 = det["x2"] / NN_SIZE
                ny2 = det["y2"] / NN_SIZE
                bx1, by1 = int(nx1 * DISP_W), int(ny1 * DISP_H)
                bx2, by2 = int(nx2 * DISP_W), int(ny2 * DISP_H)

                # ── Spatial coordinates from depth ───────────────────
                xc = yc = zc = 0.0
                if depth_frame is not None:
                    dh, dw = depth_frame.shape[:2]
                    # Map NN bbox to depth-frame coords (both aligned to CAM_A)
                    sx, sy = dw / NN_SIZE, dh / NN_SIZE
                    d_x1 = det["x1"] * sx
                    d_y1 = det["y1"] * sy
                    d_x2 = det["x2"] * sx
                    d_y2 = det["y2"] * sy

                    depth_mm = roi_median_depth(depth_frame,
                                                d_x1, d_y1, d_x2, d_y2)
                    if depth_mm > 0:
                        bb_cx = (det["x1"] + det["x2"]) / 2
                        bb_cy = (det["y1"] + det["y2"]) / 2
                        zc = depth_mm
                        xc = (bb_cx - cx_cam) * zc / fx
                        yc = (bb_cy - cy_cam) * zc / fy

                # Gravity-aligned 3D coordinates
                x3d, y3d, z3d = cam_to_3d(xc, yc, zc, pitch_rad)

                label_str = (labelMap[det["label"]]
                             if det["label"] < len(labelMap)
                             else str(det["label"]))

                # ── Draw on frame ────────────────────────────────────
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"{label_str} {det['confidence']:.2f}",
                            (bx1, by1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"3D X:{int(x3d)} Y:{int(y3d)} Z:{int(z3d)}mm",
                            (bx1, by1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"Pitch:{pitch_deg:.1f}deg",
                            (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ── HUD ──────────────────────────────────────────────────────
        cv2.putText(frame, f"Camera Pitch: {pitch_deg:.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("YOLOv8n Cube Detection", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = int(time.time())
            cv2.imwrite(f"raw_{ts}.jpg", raw_frame)
            cv2.imwrite(f"bbox_{ts}.jpg", frame)

cv2.destroyAllWindows()
