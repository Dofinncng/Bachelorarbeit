import numpy as np
import cv2

IR_Image = cv2.imread("Image_Aligment/Infrared.png")
color_image = cv2.imread("Image_Aligment/color.png")
IR_points = list()
color_points = list()

while True:
    cv2.imshow("IR_image", IR_Image)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(IR_Image, (x,y), 4, (0, 200, 0), 3)
            IR_points.append((x,y))

    cv2.setMouseCallback("IR_image", click_event)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break

IR_points = np.float32(IR_points)

while True:
    cv2.imshow("color_image", color_image)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(color_image, (x,y), 4, (0, 200, 0), 3)
            color_points.append((x,y))

    cv2.setMouseCallback("color_image", click_event)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break

color_points = np.float32(color_points)

transform_mtx = cv2.getPerspectiveTransform(color_points, IR_points)
print(transform_mtx)

corrected_color_image = cv2.warpPerspective(color_image, transform_mtx, (1920, 1080))

cv2.imshow("Neues Bild", corrected_color_image)
cv2.imshow("IR_Bild", IR_Image)
cv2.waitKey()




#if not skip:

#    array = list()
#    for i in range(len(IR_points)):
#      array.append([color_points[i][0],
#                      color_points[i][1],
#                      1,
#                      0,
#                      0,
#                      0,
#                      -color_points[i][0] * IR_points[i][0],
#                      -color_points[i][1] * IR_points[i][0]])
#        array.append([0,
#                      0,
#                      0,
#                      color_points[i][0],
#                      color_points[i][1],
#                      1,
#                      - color_points[i][0] * IR_points[i][1],
#                      - color_points[i][1] * IR_points[i][1]])

#    array = np.asarray(array)

#    array_2 = list()
#    for i in range(len(color_points)):
#        array_2.append(IR_points[i][0])
#        array_2.append(IR_points[i][1])

#    array_2 = np.asarray(array_2)
#    print(array)
#    print(array_2)

#    result = np.linalg.solve(array, array_2)
#    result = np.resize(result, (3, 3))
#    print(result)


    # M = np.array([[0.3584, 0.0089, 0], [0.0031, 0.3531, 0.0001], [0.1, 0.6311, 0.9914]])
    # print(M)
#    rotated_color_image = cv2.warpPerspective(color_image, result, (1920, 1080))
#    rotated_color_image = cv2.resize(rotated_color_image, (960, 540))

#    cv2.imshow("Ergebnis", rotated_color_image)
#    cv2.imshow("IR_Bild", IR_Image)
#    cv2.waitKey()