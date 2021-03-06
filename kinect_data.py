from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

stream = "undistored_Depth"
# raw_Depth, raw_IR, raw_Color,
# undistored_Color, undistored_Depth, undistored_IR,
# Color_Ali, IR_Ali, plot_data, RGBD

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
            full_color_frame = cv2.merge([color_frame_red,
                                          color_frame_green,
                                          color_frame_blue])
            full_color_frame = cv2.undistort(full_color_frame,
                                             mtx,
                                             dist,
                                             None,
                                             newcameramtx)
            X = cv2.getTrackbarPos("x", "KINECT Alignment Color Stream")
            Y = cv2.getTrackbarPos("y", "KINECT Alignment Color Stream")
            for i in range(20):
                full_color_frame = cv2.line(full_color_frame,
                                            (int(width / 2) + X * i, 0),
                                            (int(width / 2) + X * i, height),
                                            (100, 100, 100))
                full_color_frame = cv2.line(full_color_frame,
                                            (int(width / 2) - X * i, 0),
                                            (int(width / 2) - X * i, height),
                                            (100, 100, 100))
            for i in range(20):
                full_color_frame = cv2.line(full_color_frame,
                                            (0, int(height / 2) + Y * i),
                                            (width, int(height / 2) + Y * i),
                                            (100, 100, 100))
                full_color_frame = cv2.line(full_color_frame,
                                            (0, int(height / 2) - Y * i),
                                            (width, int(height / 2) - Y * i),
                                            (100, 100, 100))
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
            color_frame_red = np.reshape(color_frame_red,
                                         (1080, 1920))
            color_frame_green = color_frame[:, 1]
            color_frame_green = np.reshape(color_frame_green,
                                           (1080, 1920))
            color_frame_blue = color_frame[:, 2]
            color_frame_blue = np.reshape(color_frame_blue,
                                          (1080, 1920))
            full_color_frame = cv2.merge([color_frame_red,
                                          color_frame_green,
                                          color_frame_blue])
            cv2.imshow("KINECT Color Stream", full_color_frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            print(i)
            cv2.imwrite("Color_Image/distorted_image/image" + str(i) + ".png",
                        full_color_frame)
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
            depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
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
            frame = frame.astype(np.uint16)
            frame = np.uint8(frame.clip(1, 4080) / 16.)
            frame = np.reshape(frame, (424, 512))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.imshow('KINECT Infrared Stream', frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Infrared_Image/distorted_image/image" + str(i) + ".png", frame)
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
            full_color_frame = cv2.merge([color_frame_red,
                                          color_frame_green,
                                          color_frame_blue])
            undistored_full_color_frame = cv2.undistort(full_color_frame,
                                                        mtx,
                                                        dist,
                                                        None,
                                                        newcameramtx)
            together = np.concatenate((full_color_frame,
                                       undistored_full_color_frame),
                                      axis=1)
            cv2.imshow("KINECT Color Stream", full_color_frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Color_Image/distored_undistored/image" +
                        str(i) + ".png",
                        together)
            i = i + 1
        elif key == ord("d"):
            cv2.imwrite("Image_Aligment/color.png",
                        undistored_full_color_frame)


def undistored_depth(kinect_depth, mtx, dist, newcameramtx):
    i = 0
    while True:
        if kinect_depth.has_new_depth_frame():
            depth_frame = kinect_depth.get_last_depth_frame()
            # frameD = kinect_depth._depth_frame_data
            depth_frame = depth_frame.astype(np.uint8)
            depth_frame = np.reshape(depth_frame, (424, 512))
            image_depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
            #together = np.concatenate((depth_frame, undistored_depth_frame), axis=1)
            undistored_depth_frame = cv2.undistort(depth_frame, mtx, dist, None, newcameramtx)
            cv2.imshow('KINECT undistored Depth Stream', image_depth_frame)
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
            undistored_depth_frame = undistored_depth_frame[50:374, 50:462]
            cv2.imshow("frame", undistored_depth_frame)
            key = cv2.waitKey()
            while True:
                if key == 27:
                    cv2.destroyAllWindows()
                    break
            #cv2.imshow("book", cv2.applyColorMap(undistored_depth_frame[100:200, 150:300], cv2.COLORMAP_JET))
            #cv2.imshow("floor", cv2.applyColorMap(undistored_depth_frame[100:450, 30:130], cv2.COLORMAP_JET))
            #print(np.mean(undistored_depth_frame[100:200, 150:300]))
            #print(np.mean(undistored_depth_frame[100:450, 30:130]))
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            data = list()
            print(undistored_depth_frame)
            print(np.std(undistored_depth_frame))
            breakpoint()
            for y in range(len(undistored_depth_frame)):
                for x in range(len(undistored_depth_frame[y])):
                    data.append(np.array([x, 374 - y, undistored_depth_frame[y, x]]))
            data = np.asarray(data)
            print(data)
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            #ax.set_zticklabels([])
            font = "Times New Roman"
            ax.set_xlabel("X", fontname=font, fontsize=12)
            ax.set_ylabel("Y", fontname=font, fontsize=12)
            plt.rcParams["font.family"] = font
            plt.rcParams["font.size"] = 12
            #surf = ax.plot_trisurf(x, y, z, cmap="plasma", linewidth=0)
            #fig.colorbar(surf)
            #ax.xaxis.set_major_locator(MaxNLocator(5))
            #ax.yaxis.set_major_locator(MaxNLocator(5))
            #ax.zaxis.set_major_locator(MaxNLocator(5))
            ax.scatter(x, y, z, c=z, cmap='plasma', linewidth=0.5, s=0.1)
            plt.show()
            break


def undistored_infrared(kinect_infrared, mtx, dist, newcameramtx):
    i = 0
    while True:
        if kinect_infrared.has_new_infrared_frame():
            frame = kinect_infrared.get_last_infrared_frame()
            frame = frame.astype(np.uint16)
            frame = np.uint8(frame.clip(1, 4080) / 16.)
            frame = np.reshape(frame, (424, 512))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            undistored_infrared_frame = cv2.undistort(frame,
                                                      mtx,
                                                      dist,
                                                      None,
                                                      newcameramtx)
            together = np.concatenate((frame,
                                       undistored_infrared_frame),
                                      axis=1)
            cv2.imshow('KINECT undistored Infrared Stream', together)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Infrared_Image/distored_undistored/image" + str(i) + ".png",
                        together)
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
            frame = frame.astype(np.uint16)
            frame = np.uint8(frame.clip(1, 4080) / 16.)
            frame = np.reshape(frame, (424, 512))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            X = cv2.getTrackbarPos("x", "KINECT Alignment IR Stream")
            Y = cv2.getTrackbarPos("y", "KINECT Alignment IR Stream")
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
    while True:
        if kinect_depth.has_new_depth_frame():
            depth_frame = kinect_depth.get_last_depth_frame()
            depth_frame = depth_frame.astype(np.uint8)
            depth_frame = np.reshape(depth_frame, (424, 512))
            depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
            undistored_depth_frame = cv2.undistort(depth_frame, mtx, dist, None, newcameramtx)
            undistored_depth_frame = undistored_depth_frame[12:407, 15:498]
            cv2.imshow('KINECT Depth Ali Stream', depth_frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("d"):
            cv2.imwrite("Image_Aligment/depth.png", undistored_depth_frame)

def rgbd(kinect_depth,
         mtx_IR,
         dist_IR,
         newcameramtx_IR,
         kinect_color,
         mtx_color,
         dist_color,
         newcameramtx_color,
         transform_mtx,
         crop_data):

    while True:
        if kinect_depth.has_new_depth_frame():
            depth_frame = kinect_depth.get_last_depth_frame()
            # frameD = kinect_depth._depth_frame_data
            depth_frame = depth_frame.astype(np.uint8)
            depth_frame = np.reshape(depth_frame, (424, 512))
            depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
            undistored_depth_frame = cv2.undistort(depth_frame,
                                                   mtx_IR,
                                                   dist_IR,
                                                   None,
                                                   newcameramtx_IR)
            undistored_depth_frame = undistored_depth_frame[crop_data[0]: 424 - crop_data[1],
                                     crop_data[2]: 512 - crop_data[3]]

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
            undistored_full_color_frame = cv2.undistort(full_color_frame,
                                                        mtx_color,
                                                        dist_color,
                                                        None,
                                                        newcameramtx_color)
            color_image = cv2.warpPerspective(undistored_full_color_frame,
                                              transform_mtx,
                                              (512, 424))
            color_image = color_image[crop_data[0]: 424 - crop_data[1],
                          crop_data[2]: 512 - crop_data[3]]
        try:
            together = np.concatenate((color_image, undistored_depth_frame), axis=1)
            cv2.imshow("RGBD Image", together)
        except UnboundLocalError:
            pass
        key = cv2.waitKey(1)
        if key == 27:
            cv2.imwrite("Image_Aligment/RGBD_image.png", together)
            cv2.destroyAllWindows()
            break


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
    undistored_depth(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth),
                     mtx,
                     dist,
                     newcameramtx)
elif stream == "undistored_Color":
    with open("CamCalibrationData_Color", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    undistored_color(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color),
                     mtx,
                     dist,
                     newcameramtx)
elif stream == "undistored_IR":
    with open("CamCalibrationData_IR", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    undistored_infrared(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared),
                        mtx,
                        dist,
                        newcameramtx)
elif stream == "Color_Ali":
    with open("CamCalibrationData_Color", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    color_alignment(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color),
                    mtx,
                    dist,
                    newcameramtx)
elif stream == "IR_Ali":
    with open("CamCalibrationData_IR", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    IR_Ali(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared),
           mtx,
           dist,
           newcameramtx)
elif stream == "depth_ali":
    with open("CamCalibrationData_IR", "r") as file:
        mtx = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    depth_ali(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth),
              mtx,
              dist,
              newcameramtx)
elif stream == "RGBD":
    with open("CamCalibrationData_IR", "r") as file:
        mtx_IR = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist_IR = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx_IR = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    with open("CamCalibrationData_Color", "r") as file:
        mtx_color = np.loadtxt(file, skiprows=1, max_rows=3, delimiter=",")
        dist_color = np.loadtxt(file, skiprows=2, max_rows=1, delimiter=",")
        newcameramtx_color = np.loadtxt(file, skiprows=2, max_rows=3, delimiter=",")
        file.close()
    with open("Transformmatrix_color_to_infrared", "r") as file:
        transform_mtx = np.loadtxt(file, skiprows=1, delimiter=",")
    crop_data = list()
    with open("Crop_Image_data.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            try:
                line = int(line)
                crop_data.append(line)
            except ValueError:
                pass
    rgbd(PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth),
         mtx_IR,
         dist_IR,
         newcameramtx_IR,
         PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color),
         mtx_color,
         dist_color,
         newcameramtx_color,
         transform_mtx,
         crop_data)

