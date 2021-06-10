#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
from multiprocessing import Process, Queue
import cv2
import depthai as dai
import sys
import numpy as np
import time

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default="data\\14442C10E162E5D200", type=str, help="Path where to store the captured data")
parser.add_argument('-d', '--depth', action='store_true', default=False, help="Use saved depth maps")
args = parser.parse_args()

# Get the stored frames path
dest = Path(args.path).resolve().absolute()
frames = os.listdir(str(dest))
# TODO: if not int parsable, skip
frames_sorted = sorted([int(i) for i in frames])

class Replay:
    def __init__(self, path, device):
        self.path = path
        self.mono_size = self.__get_mono_size()
        self.color_size = self.__get_color_size()
        self.device = device

    def create_input_queues(self):
        # Create input queues
        inputs = ["rgbIn"]
        if args.depth:
            inputs.append("depthIn")
        else: # Use mono frames
            inputs.append("left")
            inputs.append("right")
        self.q = {}
        for input_name in inputs:
            self.q[input_name] = self.device.getInputQueue(input_name)

    def __get_color_size(self):
        files = self.get_files(0)
        for file in files:
            if not file.startswith("color"): continue
            frame = self.read_color(self.get_path(0, file))
            return frame.shape
        return None

    def __get_mono_size(self):
        files = self.get_files(0)
        for file in files:
            if not file.startswith("left"): continue
            frame = self.read_mono(self.get_path(0, file))
            return frame.shape
        return None

    def to_planar(self, arr, shape):
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    def read_color(self,path):
        return cv2.imread(path)
    def read_mono(self,path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    def read_depth(self,path):
        with open(path, "rb") as depth_file:
            return list(depth_file.read())
    def get_path(self, folder, file = None):
        if file is None:
            return str((Path(self.path) / str(folder)).resolve().absolute())

        return str((Path(self.path) / str(folder) / file).resolve().absolute())

    def read_files(self, frame_folder, files):
        frames = {}
        for file in files:
            file_path = self.get_path(frame_folder, file)
            if file.startswith("right") or file.startswith("left"):
                frame = self.read_mono(file_path)
            elif file.startswith("depth"):
                frame = self.read_depth(file_path)
            elif file.startswith("color"):
                frame = self.read_color(file_path)
            else:
                # print(f"Unknown file found! {file}")
                continue
            frames[os.path.splitext(file)[0]] = frame
        return frames

    def get_files(self, frame_folder):
        return os.listdir(self.get_path(frame_folder))

    def send_frames(self, images):
        for name, img in images.items():
            print(f"sending frame {name}")
            replay.send_frame(name, img)

    # Send recorded frames from the host to the depthai
    def send_frame(self, name, frame):
        if name in ["left", "right"] and not args.depth:
            self.send_mono(frame, name)
        elif name == "depth" and args.depth:
            self.send_depth(frame)
        elif name == "color":
            self.send_rgb(frame)

    def send_mono(self, img, name):
        h, w = img.shape
        frame = dai.ImgFrame()
        frame.setData(cv2.flip(img, 1)) # Flip the rectified frame
        frame.setType(dai.RawImgFrame.Type.RAW8)
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum((2 if name == "right" else 1))
        self.q[name].send(frame)
    def send_rgb(self, img):
        preview = img[0:1080, 420:1500] # Crop before sending
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setData(self.to_planar(preview, (300, 300)))
        frame.setWidth(300)
        frame.setHeight(300)
        frame.setInstanceNum(0)
        self.q["rgbIn"].send(frame)
    def send_depth(self, depth):
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.RAW16)
        frame.setData(depth)
        frame.setWidth(640)
        frame.setHeight(400)
        frame.setInstanceNum(0)
        self.q["depthIn"].send(frame)


def create_bird_frame():
    fov = 68.3
    frame = np.zeros((300, 100, 3), np.uint8)
    cv2.rectangle(frame, (0, 283), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

    alpha = (180 - fov) / 2
    center = int(frame.shape[1] / 2)
    max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
    fov_cnt = np.array([
        (0, frame.shape[0]),
        (frame.shape[1], frame.shape[0]),
        (frame.shape[1], max_p),
        (center, frame.shape[0]),
        (0, max_p),
        (0, frame.shape[0]),
    ])
    cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
    return frame

def draw_bird_frame(frame, y, z, id = None):
    global MAX_Z
    max_y = 2000 #mm
    pointY = frame.shape[0] - int(z / (MAX_Z - 10000) * frame.shape[0]) - 20
    pointX = int(y / max_y * frame.shape[1] + frame.shape[1]/2)
    # print(f"Y {y}, Z {z} - Birds: X {pointX}, Y {pointY}")
    if id is not None:
        cv2.putText(frame, str(id), (pointX - 30, pointY + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
    cv2.circle(frame, (pointX, pointY), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)


# Draw spatial detections / tracklets to the frame
def display_spatials(frame, detections, name, tracker = False):
    color = (255, 207, 17) if name == "depth" else (30, 211, 255)
    h = frame.shape[0]
    w = frame.shape[1]
    birdFrame = create_bird_frame()
    for detection in detections:
        # Denormalize bounding box
        imgDet = detection.srcImgDetection if tracker else detection
        x1 = int(imgDet.xmin * w)
        x2 = int(imgDet.xmax * w)
        y1 = int(imgDet.ymin * h)
        y2 = int(imgDet.ymax * h)

        try:
            label = labelMap[detection.label]
        except:
            label = detection.label


        if tracker:
           # if crashAvoidance.remove_lost_tracklet(detection): continue
            # If these are tracklets, display ID as well
            cv2.putText(frame, f"Car {detection.id}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            # frame = crashAvoidance.drawArrows(frame, detection)
            # cv2.putText(frame, detection.status.name, (x1 + 10, y1 - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            # cv2.putText(frame, "{:.2f}".format(detection.srcImgDetection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            # speed = crashAvoidance.calculate_speed(detection)

            # cv2.putText(frame, "{:.1f} km/h".format(speed), (x1 + 10, y1 - 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        # else:
            # cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        # cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
        # If vehicle is too far away, coordinate estimation is off so don't display them
        if (x2-x1)*(y2-y1) < 600: continue
        draw_bird_frame(birdFrame, detection.spatialCoordinates.y, detection.spatialCoordinates.z, detection.id if tracker else None)
        cv2.putText(frame, "X: {:.1f} m".format(int(detection.spatialCoordinates.x) / 1000.0), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "Y: {:.1f} m".format(int(detection.spatialCoordinates.y) / 1000.0), (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "Z: {:.1f} m".format(int(detection.spatialCoordinates.z) / 1000.0), (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
    if args.birdsview: return birdFrame


# Create the pipeline
def create_pipeline(replay):
    pipeline = dai.Pipeline()

    rgb_in = pipeline.createXLinkIn()
    rgb_in.setStreamName("rgbIn")

    if not args.depth:
        left_in = pipeline.createXLinkIn()
        right_in = pipeline.createXLinkIn()
        left_in.setStreamName("left")
        right_in.setStreamName("right")

        stereo = pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(240)
        median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
        stereo.setMedianFilter(median)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)
        mono_size = replay.mono_size
        stereo.setInputResolution(mono_size[1], mono_size[0])
        # Since frames are already rectified
        stereo.setEmptyCalibration()

        left_in.out.link(stereo.left)
        right_in.out.link(stereo.right)

        right_s_out = pipeline.createXLinkOut()
        right_s_out.setStreamName("rightS")
        stereo.syncedRight.link(right_s_out.input)

        left_s_out = pipeline.createXLinkOut()
        left_s_out.setStreamName("leftS")
        stereo.syncedLeft.link(left_s_out.input)

    spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
    spatialDetectionNetwork.setBlobPath("models/mobilenet-ssd_openvino_2021.2_6shave.blob")
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.3)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    if args.depth:
        depth_in = pipeline.createXLinkIn()
        depth_in.setStreamName("depthIn")
        depth_in.out.link(spatialDetectionNetwork.inputDepth)
    else:
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

    rgb_in.out.link(spatialDetectionNetwork.input)

    bbOut = pipeline.createXLinkOut()
    bbOut.setStreamName("bb")
    spatialDetectionNetwork.boundingBoxMapping.link(bbOut.input)

    detOut = pipeline.createXLinkOut()
    detOut.setStreamName("det")
    spatialDetectionNetwork.out.link(detOut.input)

    depthOut = pipeline.createXLinkOut()
    depthOut.setStreamName("depth")
    spatialDetectionNetwork.passthroughDepth.link(depthOut.input)

    rgbOut = pipeline.createXLinkOut()
    rgbOut.setStreamName("rgb")
    spatialDetectionNetwork.passthrough.link(rgbOut.input)

    return pipeline

# Pipeline defined, now the device is connected to
with dai.Device() as device:
    replay = Replay(path=args.path, device=device)
    device.startPipeline(create_pipeline(replay))
    replay.create_input_queues()

    if not args.depth:
        qLeftS = device.getOutputQueue(name="leftS", maxSize=4, blocking=False)
        qRightS = device.getOutputQueue(name="rightS", maxSize=4, blocking=False)

    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    qBb = device.getOutputQueue(name="bb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="det", maxSize=4, blocking=False)
    qRgbOut = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    color = (255, 0, 0)
    # if args.tracker:
    #     qTracklets = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)
    # Read rgb/mono frames, send them to device and wait for the spatial object detection results
    for frame_folder in frames_sorted:
        files = replay.get_files(frame_folder)

        # Read the frames from the FS
        images = replay.read_files(frame_folder, files)

        replay.send_frames(images)
        # Send first frames twice for first iteration (depthai FW limitation)
        if frame_folder == 0: # TODO: debug
            replay.send_frames(images)
            replay.send_frames(images)
            replay.send_frames(images)

        inRgb = qRgbOut.get()
        rgbFrame = inRgb.getCvFrame().reshape((300, 300, 3))

        if not args.depth:
            leftS = qLeftS.get().getCvFrame()
            rightS = qRightS.get().getCvFrame()
            cv2.imshow("left", leftS)
            cv2.imshow("right", rightS)

        def get_colored_depth(frame):
            depthFrameColor = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            return cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        depthFrameColor = get_colored_depth(qDepth.get().getFrame())
        # cv2.imshow("replayed depth", depthFrameColor)
        # cv2.imshow("recorded depth", get_colored_depth(np.array(images["depth"]).astype(np.uint8).view(np.uint16).reshape(replay.mono_size)))

        height = inRgb.getHeight()
        width = inRgb.getWidth()


        #inDet = qDet.tryGet()
        #if inDet is not None:
        #    if len(inDet.detections) != 0:
                # Display boundingbox mappings on the depth frame
        #        bbMapping = qBb.get()
        #        roiDatas = bbMapping.getConfigData()
        #        for roiData in roiDatas:
        #            roi = roiData.roi
        #            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
        #            topLeft = roi.topLeft()
        #            bottomRight = roi.bottomRight()
        #            xmin = int(topLeft.x)
        #            ymin = int(topLeft.y)
        #            xmax = int(bottomRight.x)
        #            ymax = int(bottomRight.y)
        #            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), (0,255,0), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # Display (spatial) object detections on the color frame
        #    for detection in inDet.detections:
                # Denormalize bounding box
        #        x1 = int(detection.xmin * 300)
        #        x2 = int(detection.xmax * 300)
        #        y1 = int(detection.ymin * 300)
        #        y2 = int(detection.ymax * 300)
        #        try:
        #            label = labelMap[detection.label]
        #        except:
        #            label = detection.label
        #        cv2.putText(rgbFrame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        #        cv2.putText(rgbFrame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        #        cv2.putText(rgbFrame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        #        cv2.putText(rgbFrame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        #        cv2.putText(rgbFrame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        #        cv2.rectangle(rgbFrame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(rgbFrame, str(frame_folder), (10, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        #rough mark: cv2.line(depthFrameColor, (300, 250), (700, 250), (0, 255, 0), 3)
        #centerline
        cv2.line(depthFrameColor, (640, 0), (640, 255), (0, 255, 0), 3)
        # road line
        array_of_dist_lengths = []
        for y in range(720, 255, -1):
            dist_y = depthFrameColor[640, y]
        #    print(dist_y)

        cv2.imshow("rgb", rgbFrame)
        cv2.imshow("depth", depthFrameColor)


        inRgb = qRgbOut.tryGet()
        if inRgb is not None:
            rgbFrame = inRgb.getCvFrame().reshape((model_height, model_width, 3))

            def get_colored_depth(frame):
                frame = replay.crop_frame(frame)
                depthFrameColor = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                return cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
            depthFrameColor = get_colored_depth(qDepth.get().getFrame())

            if args.tracker: detections = qDet.get().tracklets
            else: detections = qDet.get().detections

            birdsView = display_spatials(rgbFrame, detections, "color", args.tracker)
            def save_png(folder, name, item):
                frames_path = Path(IMG_SAVE_PATH) / str(name)
                frames_path.mkdir(parents=True, exist_ok=True)
                if folder < 10: folder = "000" + str(folder)
                elif folder < 100: folder = "00" + str(folder)
                elif folder < 1000: folder = "0" + str(folder)
                cv2.imwrite(str(frames_path / f"{folder}.png"), item)

            h = rgbFrame.shape[0]
            w = rgbFrame.shape[1]
            if not args.depth and args.monos:
                leftS = qLeftS.get().getCvFrame()
                rightS = qRightS.get().getCvFrame()
                left = cv2.resize(leftS, (w,h))
                right = cv2.resize(rightS, (w,h))
                cv2.imshow("left", left)
                cv2.imshow("right", right)
                save_png(frame_folder, "left", left)
                save_png(frame_folder, "right", right)
            if args.monohost:
                cv2.imshow("left", cv2.resize(images["left"], (w,h)))
                cv2.imshow("right", cv2.resize(images["right"], (w,h)))

            cv2.imshow("rgb", rgbFrame)
            depthFrameColor = cv2.resize(depthFrameColor, (w,h))
            display_spatials(depthFrameColor, detections, "depth", args.tracker)
            cv2.imshow("depth", depthFrameColor)
            save_png(frame_folder, "rgb", rgbFrame)
            save_png(frame_folder, "depth", depthFrameColor)
            save_png(frame_folder, "birdsview", birdsView)
        if cv2.waitKey(1) == ord('q'):
            break

