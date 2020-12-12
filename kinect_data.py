from pykinect2 import PyKinectV2
# from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

stream = "depth_ali"


# raw_Depth, raw_IR, raw_Color, undistored_Color, undistored_Depth, undistored_IR, Color_Ali, IR_Ali, plot_data


def color_alignment(kinect_color, mtx, dist, newcameramtx):
    cv2.namedWindow("KINECT Alignment Color Stream")
    cv2.createTrackbar("x", "KINECT Alignment Color Stream", 0, 120, lambda x: None)
    cv2.createTrackbar("y", "KINECT Alignment Color Stream", 0, 120, lambda y: None)

    height = 1080
    width = 1920
    while True:
        if kinect_color.has_new_color_frame():

            color_frame = kinect_color.get_last_color_frame()

            color_frame = np.reshape(color_frame, (2073600, 4))
            color_frame = color_frame[:, 0:3]

            # extract then combine the RBG data
            color_frame_red = color_frame[:, 0]
            color_frame_red = np.reshape(color_frame_red, (1080, 1920))
            color_frame_green = color_frame[:, 1]
            color_frame_green = np.reshape(color_frame_green, (1080, 1920))
            color_frame_blue = color_frame[:, 2]
            color_frame_blue = np.reshape(color_frame_blue, (1080, 1920))
            full_color_frame = cv2.merge([color_frame_red, color_frame_green, color_frame_blue])
            # full_color_frame = cv2.line(full_color_frame, (0, int(1080/2)), (1920, int(1080/2)), (0, 0, 0), 2)
            # full_color_frame = cv2.line(full_color_frame, (0, 1920 / 2), (1080, 1920 / 2), (0, 0, 0), 2)

            full_color_frame = cv2.undistort(full_color_frame, mtx, dist, None, newcameramtx)

            X = cv2.getTrackbarPos("x", "KINECT Alignment Color Stream")
            Y = cv2.getTrackbarPos("y", "KINECT Alignment Color Stream")

            # line_distance_x = width / X
            # line_distance_y = height / Y

            for i in range(20):
                full_color_frame = cv2.line(full_color_frame, (int(width / 2) + X * i, 0),
                                            (int(width / 2) + X * i, height), (100, 100, 100))
                full_color_frame = cv2.line(full_color_frame, (int(width / 2) - X * i, 0),
                                            (int(width / 2) - X * i, height), (100, 100, 100))
            for i in range(20):
                full_color_frame = cv2.line(full_color_frame, (0, int(height / 2) + Y * i),
                                            (width, int(height / 2) + Y * i), (100, 100, 100))
                full_color_frame = cv2.line(full_color_frame, (0, int(height / 2) - Y * i),
                                            (width, int(height / 2) - Y * i), (100, 100, 100))

            cv2.imshow("KINECT Alignment Color Stream", full_color_frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break


def raw_color_stream(kinect_color):
    i = 0
    while True:
        if kinect_color.has_new_color_frame():
            color_frame = kinect_color.get_last_color_frame()

            color_frame = np.reshape(color_frame, (2073600, 4))
            color_frame = color_frame[:, 0:3]

            # extract then combine the RBG data
            color_frame_red = color_frame[:, 0]
            color_frame_red = np.reshape(color_frame_red, (1080, 1920))
            color_frame_green = color_frame[:, 1]
            color_frame_green = np.reshape(color_frame_green, (1080, 1920))
            color_frame_blue = color_frame[:, 2]
            color_frame_blue = np.reshape(color_frame_blue, (1080, 1920))
            full_color_frame = cv2.merge([color_frame_red, color_frame_green, color_frame_blue])
            cv2.imshow("KINECT Color Stream", full_color_frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            print(i)
            cv2.imwrite("Color_Image/image" + str(i) + ".png", full_color_frame)
            i = i + 1


def raw_depth_stream(kinect_depth):
    i = 0
    cv2.namedWindow("KINECT Depth Stream")
    cv2.createTrackbar("low", "KINECT Depth Stream", 0, 255, lambda x: None)
    cv2.createTrackbar("high", "KINECT Depth Stream", 0, 255, lambda x: None)
    cv2.setTrackbarPos("high", "KINECT Depth Stream", 255)
    while True:
        if kinect_depth.has_new_depth_frame():
            depth_frame = kinect_depth.get_last_depth_frame()
            # frameD = kinect_depth._depth_frame_data
            depth_frame = depth_frame.astype(np.uint8)
            depth_frame = np.reshape(depth_frame, (424, 512))
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)

            low = cv2.getTrackbarPos("low", "KINECT Depth Stream")
            high = cv2.getTrackbarPos("high", "KINECT Depth Stream")
            low = np.array([low, low, low])
            high = np.array([high, high, high])

            mask = cv2.inRange(depth_frame, low, high)
            depth_frame = cv2.bitwise_and(depth_frame, depth_frame, mask=mask)

            cv2.imshow('KINECT Depth Stream', depth_frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Depth_Image/image" + str(i) + ".png", depth_frame)
            i = i + 1


def raw_infrared_stream(kinect_infrared):
    i = 0
    while True:
        if kinect_infrared.has_new_infrared_frame():
            frame = kinect_infrared.get_last_infrared_frame()
            frame = frame.astype(np.uint16)  # infrared frame是uint16类型
            frame = np.uint8(frame.clip(1, 4080) / 16.)  # 转换为uint8时，需要避免溢出255*16=4080
            frame = np.reshape(frame, (424, 512))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.imshow('KINECT Infrared Stream', frame)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Infrared_Image/image" + str(i) + ".png", frame)
            i = i + 1


def undistored_color(kinect_color, mtx, dist, newcameramtx):
    i = 0
    while True:
        if kinect_color.has_new_color_frame():
            color_frame = kinect_color.get_last_color_frame()

            color_frame = np.reshape(color_frame, (2073600, 4))
            color_frame = color_frame[:, 0:3]

            # extract then combine the RBG data
            color_frame_red = color_frame[:, 0]
            color_frame_red = np.reshape(color_frame_red, (1080, 1920))
            color_frame_green = color_frame[:, 1]
            color_frame_green = np.reshape(color_frame_green, (1080, 1920))
            color_frame_blue = color_frame[:, 2]
            color_frame_blue = np.reshape(color_frame_blue, (1080, 1920))
            full_color_frame = cv2.merge([color_frame_red, color_frame_green, color_frame_blue])

            undistored_full_color_frame = cv2.undistort(full_color_frame, mtx, dist, None, newcameramtx)

            together = np.concatenate((full_color_frame, undistored_full_color_frame), axis=1)
            cv2.imshow("KINECT Color Stream", full_color_frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Color_Image/distored_undistored/image" + str(i) + ".png", together)
            i = i + 1
        elif key == ord("d"):
            cv2.imwrite("Image_Aligment/color.png", undistored_full_color_frame)


def undistored_depth(kinect_depth, mtx, dist, newcameramtx):
    i = 0
    while True:
        if kinect_depth.has_new_depth_frame():
            depth_frame = kinect_depth.get_last_depth_frame()
            # frameD = kinect_depth._depth_frame_data
            depth_frame = depth_frame.astype(np.uint8)
            depth_frame = np.reshape(depth_frame, (424, 512))
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)

            undistored_depth_frame = cv2.undistort(depth_frame, mtx, dist, None, newcameramtx)
            undistored_depth_frame = undistored_depth_frame[55:378, 37:466]
            # undistored_depth_frame = undistored_depth_frame[12:407, 15:498]

            # together = np.concatenate((depth_frame, undistored_depth_frame), axis=1)
            cv2.imshow('KINECT undistored Depth Stream', undistored_depth_frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Depth_Image/distored_undistored/image" + str(i) + ".png", together)
            i = i + 1
        elif key == ord("d"):
            cv2.imwrite("Image_Aligment/depth.png", undistored_depth_frame)
        elif key == ord("p"):
            # undistored_depth_frame = undistored_depth_frame[50:374, 50:462]
            # print(undistored_depth_frame)
            cv2.destroyAllWindows()
            data = list()
            for y in range(len(undistored_depth_frame)):
                for x in range(len(undistored_depth_frame[y])):
                    if 0 < undistored_depth_frame[y, x][0] < 255:
                        data.append(np.array([x, 363 - y, 255 - undistored_depth_frame[y, x][0]]))

            data = np.asarray(data)
            print(data)

            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            fig = plt.figure()

            ax = fig.add_subplot(111, projection="3d")
            ax.set_zlim(0, 255)
            plt.ylabel("Y")
            plt.xlabel("X")

            surf = ax.plot_trisurf(x, y, z, cmap="plasma", linewidth=0)
            fig.colorbar(surf)

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.zaxis.set_major_locator(MaxNLocator(5))

            ax.view_init(90, -90)

            plt.show()
            break


def undistored_infrared(kinect_infrared, mtx, dist, newcameramtx):
    i = 0
    while True:
        if kinect_infrared.has_new_infrared_frame():
            frame = kinect_infrared.get_last_infrared_frame()
            frame = frame.astype(np.uint16)  # infrared frame是uint16类型
            frame = np.uint8(frame.clip(1, 4080) / 16.)  # 转换为uint8时，需要避免溢出255*16=4080
            frame = np.reshape(frame, (424, 512))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            undistored_infrared_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

            together = np.concatenate((frame, undistored_infrared_frame), axis=1)
            cv2.imshow('KINECT undistored Infrared Stream', together)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Infrared_Image/distored_undistored/image" + str(i) + ".png", together)
            i = i + 1
        elif key == ord("d"):
            cv2.imwrite("Image_Aligment/Infrared.png", undistored_infrared_frame)


def IR_Ali(kinect_infrared, mtx, dist, newcameramtx):
    cv2.namedWindow("KINECT Alignment IR Stream")
    cv2.createTrackbar("x", "KINECT Alignment IR Stream", 0, 120, lambda x: None)
    cv2.createTrackbar("y", "KINECT Alignment IR Stream", 0, 120, lambda x: None)

    height = 424
    width = 512
    i = 0

    while True:
        if kinect_infrared.has_new_infrared_frame():
            frame = kinect_infrared.get_last_infrared_frame()
            frame = frame.astype(np.uint16)  # infrared frame是uint16类型
            frame = np.uint8(frame.clip(1, 4080) / 16.)  # 转换为uint8时，需要避免溢出255*16=4080
            frame = np.reshape(frame, (424, 512))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

            X = cv2.getTrackbarPos("x", "KINECT Alignment IR Stream")
            Y = cv2.getTrackbarPos("y", "KINECT Alignment IR Stream")

            # line_distance_x = width / X
            # line_distance_y = height / Y

            for i in range(20):
                frame = cv2.line(frame, (int(width / 2) + X * i, 0),
                                 (int(width / 2) + X * i, height), (100, 100, 100))
                frame = cv2.line(frame, (int(width / 2) - X * i, 0),
                                 (int(width / 2) - X * i, height), (100, 100, 100))
            for i in range(20):
                frame = cv2.line(frame, (0, int(height / 2) + Y * i),
                                 (width, int(height / 2) + Y * i), (100, 100, 100))
                frame = cv2.line(frame, (0, int(height / 2) - Y * i),
                                 (width, int(height / 2) - Y * i), (100, 100, 100))

            # together = np.concatenate((frame, undistored_infrared_frame), axis=1)
            cv2.imshow("KINECT Alignment IR Stream", frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Image_Aligment/Aligment_to_floor/Image" + str(i) + ".png", frame)
            i = i + 1

def depth_ali(kinect_depth, mtx, dist, newcameramtx):
    cv2.namedWindow('KINECT Depth Ali Stream')
    cv2.createTrackbar("x", 'KINECT Depth Ali Stream', 1, 10, lambda x: None)
    i = 0
    while True:
        if kinect_depth.has_new_depth_frame():
            depth_frame = kinect_depth.get_last_depth_frame()
            # frameD = kinect_depth._depth_frame_data
            depth_frame = depth_frame.astype(np.uint8)

            # factor = cv2.getTrackbarPos("x", "Kinect Depth Ali Stream")
            # depth_frame = factor * depth_frame

            depth_frame = np.reshape(depth_frame, (424, 512))
            # depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)

            depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

            undistored_depth_frame = cv2.undistort(depth_frame, mtx, dist, None, newcameramtx)
            # undistored_depth_frame = undistored_depth_frame[55:378, 37:466]
            undistored_depth_frame = undistored_depth_frame[12:407, 15:498]


            # together = np.concatenate((depth_frame, undistored_depth_frame), axis=1)
            cv2.imshow('KINECT Depth Ali Stream', depth_frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("d"):
            cv2.imwrite("Image_Aligment/depth.png", undistored_depth_frame)


if stream == "raw_Depth":
    raw_depth_stream(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth))

elif stream == "raw_IR":
    raw_infrared_stream(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared))

elif stream == "raw_Color":
    raw_color_stream(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color))

elif stream == "undistored_Depth":
    with open("CamCalibrationData_IR", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    undistored_depth(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth), mtx, dist, newcameramtx)

elif stream == "undistored_Color":
    with open("CamCalibrationData_Color", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    undistored_color(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color), mtx, dist, newcameramtx)

elif stream == "undistored_IR":
    with open("CamCalibrationData_IR", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    undistored_infrared(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared), mtx, dist, newcameramtx)

elif stream == "Color_Ali":
    with open("CamCalibrationData_Color", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()

    color_alignment(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color), mtx, dist, newcameramtx)

elif stream == "IR_Ali":
    with open("CamCalibrationData_IR", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    IR_Ali(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared), mtx, dist, newcameramtx)

elif stream == "depth_ali":
    with open("CamCalibrationData_IR", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    depth_ali(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth), mtx, dist, newcameramtx)

################################yyyyy####xxxxxx
# fitted_color_image = color_image[0:1080, 270:1750]
# depth_image  = depth_image[4:424, 2:510]
# fitted_color_image = color_image
# fitted_color_image = cv2.resize(fitted_color_image, (508, 420))


# print(depth_image.shape, fitted_color_image.shape)#

# combinded_image = np.concatenate((depth_image,fitted_color_image), axis=1)

# dst = cv2.addWeighted(depth_image, 1, fitted_color_image, 1, 0)
# cv2.imshow("nebeneinander", combinded_image)
# cv2.imshow("Ueberlagerung", dst)
# cv2.waitKey()

# breakpoint()

# SCALE_FACTOR = 0.25
# fitted_color_image_width, fitted_color_image_height, channels = fitted_color_image.shape
# if fitted_color_image_width%(1/SCALE_FACTOR) or fitted_color_image_height%(1/SCALE_FACTOR):
#    print("Fehlermeldung")
# fitted_color_image = cv2.resize(fitted_color_image,
#                             (int(fitted_color_image_height * SCALE_FACTOR), int(fitted_color_image_width * SCALE_FACTOR)))


# cv2.imshow("Farbbild", fitted_color_image)
# cv2.waitKey()
# cv2.imshow("Tiefenbild", depth_image)
# cv2.waitKey()
