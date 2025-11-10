#!/usr/bin/env python3

"""
Node providing HUMAN_BODY_POSE messages based on a Intel RealSense camera.
"""

import argparse
import cv2
from math import nan
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from pymavlink import mavutil
import pyrealsense2 as rs
from time import time
from typing import Optional, Tuple

import mavlink_all as mavlink


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # fmt: off
    rs_group = parser.add_argument_group("RealSense")
    _ = rs_group.add_argument("--resolution", default=[640, 480], type=int, nargs=2, metavar=('width', 'height'),
                              help="Resolution of the realsense stream")
    _ = rs_group.add_argument("-f", "--fps", default=30, type=int,
                              help="Framerate of the realsense stream")

    mp_group = parser.add_argument_group("MediaPipe")
    _ = mp_group.add_argument("--model-complexity", default=1, type=int,
                              help="Set model complexity (0=Light, 1=Full, 2=Heavy).")
    _ = mp_group.add_argument("--no-smooth-landmarks", action="store_false", help="Disable landmark smoothing")
    _ = mp_group.add_argument("--static-image-mode", action="store_true", help="Enables static image mode")
    _ = mp_group.add_argument("--min-detection-confidence", type=float, default=0.5,
                              help="Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful")
    _ = mp_group.add_argument("--min-tracking-confidence", type=float, default=0.5,
                              help=" Minimum confidence value ([0.0, 1.0]) to be considered tracked successfully")

    marsh_group = parser.add_argument_group("MARSH")
    _ = marsh_group.add_argument("-m", "--manager",
                                 help="MARSH Manager IP address", default="127.0.0.1")
    _ = marsh_group.add_argument("--one-point", type=int,
                                 help="send only tracked point with this index")
    # fmt: on
    args = parser.parse_args()

    # setup camera loop
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        smooth_landmarks=args.no_smooth_landmarks,
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # create realsense pipeline
    pipeline = rs.pipeline()

    width, height = args.resolution

    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)

    profile = pipeline.start(config)

    prev_frame_time = 0

    # create MAVLink connection
    connection_string = f"udpout:{args.manager}:24400"
    mav = mavlink.MAVLink(mavutil.mavlink_connection(connection_string))
    mav.srcSystem = 1  # default system
    mav.srcComponent = mavlink.MAV_COMP_ID_USER1 + (
        mavlink.MARSH_TYPE_EYE_TRACKER - mavlink.MARSH_TYPE_MANAGER
    )
    print(f"Sending to {connection_string}")

    # controlling when messages should be sent
    heartbeat_next = 0.0
    heartbeat_interval = 1.0

    # monitoring connection to manager with heartbeat
    timeout_interval = 5.0
    manager_timeout = 0.0
    manager_connected = False

    try:
        while True:
            if time() >= heartbeat_next:
                mav.heartbeat_send(
                    mavlink.MARSH_TYPE_EYE_TRACKER,
                    mavlink.MAV_AUTOPILOT_INVALID,
                    mavlink.MAV_MODE_FLAG_TEST_ENABLED,
                    0,
                    mavlink.MAV_STATE_ACTIVE,
                )
                heartbeat_next = time() + heartbeat_interval

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            if not color_frame:
                break

            image = np.asanyarray(color_frame.get_data())

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # alpha=0.03 is lossless, but we scale it to saturate at 2 meters to have more legible colors in RPC cockpit
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.125), cv2.COLORMAP_JET
            )

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = image.shape

            # send the pose to MARSH
            send_pose(
                profile,
                mav,
                round(color_frame.get_timestamp() * 1000),
                results.pose_landmarks,
                depth_image,
                args.min_tracking_confidence,
                args.one_point,
            )

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                depth_colormap, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # If depth and color resolutions are different, resize depth image to match color image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_depth_image = cv2.resize(
                    depth_colormap,
                    dsize=(color_colormap_dim[1], color_colormap_dim[0]),
                    interpolation=cv2.INTER_AREA,
                )
                images = np.vstack((image, resized_depth_image))
            else:
                images = np.vstack((image, depth_colormap))

            current_time = time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time

            # Mirror-like display
            images = cv2.flip(images, 1)
            cv2.putText(
                images,
                "FPS: %.0f" % fps,
                (7, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("RealSense Pose Detector (Esc to quit)", images)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            # handle incoming MAVLink messages
            try:
                while (message := mav.file.recv_msg()) is not None:
                    message: mavlink.MAVLink_message
                    if message.get_type() == "HEARTBEAT":
                        heartbeat: mavlink.MAVLink_heartbeat_message = message
                        if heartbeat.type == mavlink.MARSH_TYPE_MANAGER:
                            if not manager_connected:
                                print("Connected to simulation manager")
                            manager_connected = True
                            manager_timeout = time() + timeout_interval
            except ConnectionResetError:
                # thrown on Windows when there is no peer listening
                pass

            if manager_connected and time() > manager_timeout:
                manager_connected = False
                print("Lost connection to simulation manager")
    finally:
        pose.close()
        pipeline.stop()


def send_pose(
    profile,
    mav,
    time_usec: int,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    depth_image,
    tracking_confidence: float = 0.5,
    one_point: Optional[int] = None,
):
    if landmark_list is None:
        return

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    for point, landmark in enumerate(landmark_list.landmark):
        # Only send visible points
        if not landmark.visibility > tracking_confidence:
            continue

        if one_point is not None and point != one_point:
            continue

        x, y, z = deproject(
            depth_intrinsics, depth_scale, depth_image, landmark.x, landmark.y
        )

        mav.human_body_pose_send(
            time_usec,
            0,  # default sensor
            point,
            x if not x == 0.0 else nan,  # if depth is zero, invalid projection
            y if not x == 0.0 else nan,  # if depth is zero, invalid projection
            z if not x == 0.0 else nan,  # if depth is zero, invalid projection
            landmark.x,
            landmark.y,
        )


def deproject(
    intrinsics,
    depth_scale: float,
    depth_image: np.ndarray,
    fraction_x: float,
    fraction_y: float,
) -> Tuple[float, float, float]:
    x = max(0, min(intrinsics.width - 1, round(fraction_x * intrinsics.width)))
    y = max(0, min(intrinsics.height - 1, round(fraction_y * intrinsics.height)))
    depth = depth_scale * depth_image[y][x]

    result = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    # result[0]: right, result[1]: down, result[2]: forward
    return result[2], result[0], result[1]


if __name__ == "__main__":
    main()
