import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*5,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Color_Image/*.png')
i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,5),None)

    # If found, add object points, image points (after refining them)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,5), corners2,ret)
        cv2.imwrite("Color_Image/Checkerboard_Image/Image" + str(i) + ".png", img)
        cv2.waitKey(1)
        i = i + 1


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#img = cv2.imread('image7.png')
# h,  w = img.shape[:2]
w, h = 1920, 1080
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
# img = cv2.imread("Color_Image/image0.png")
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imshow("normal", img)
# cv2.imshow('Calibration',dst)
# cv2.waitKey()


with open("CamCalibrationData_Color", "w") as file:
    file.write("mtx\n")
    np.savetxt(file, mtx, delimiter=",")
    file.write("\ndist \n")
    np.savetxt(file, dist, delimiter=",")
    file.write("\nnewcameramtx \n")
    np.savetxt(file, newcameramtx, delimiter=",")
    file.write("\nroi \n")
    np.savetxt(file, roi, delimiter=",")
    file.close()

