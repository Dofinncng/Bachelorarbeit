import cv2
import numpy as np

ir_image = cv2.imread("Image_Aligment/Perspective/IR_original.png")

color_image = cv2.imread("Image_Aligment/Perspective/color_new.png")

color_edge = cv2.Canny(color_image, 200, 300)
ir_edge = cv2.Canny(ir_image, 100, 200)

blend_image = cv2.addWeighted(color_edge, 0.5, ir_edge, 0.5, 0)
blend_image = cv2.cvtColor(blend_image, cv2.COLOR_GRAY2RGB)

LINE_THICKNESS = 2

def trackbar_top(x):
    display_image = blend_image.copy()
    display_image = cv2.line(display_image, (0, x), (512, x), (255, 0, 0), LINE_THICKNESS)

    bottom = cv2.getTrackbarPos("Bottom", "Crop Images")
    display_image = cv2.line(display_image, (0, 424 - bottom), (512, 424 - bottom), (255, 0, 0), LINE_THICKNESS)

    left = cv2.getTrackbarPos("Left", "Crop Images")
    display_image = cv2.line(display_image, (left, 0), (left, 424), (255, 0, 0), LINE_THICKNESS)

    right = cv2.getTrackbarPos("Right", "Crop Images")
    display_image = cv2.line(display_image, (512 - right, 0), (512 - right, 424), (255, 0, 0), LINE_THICKNESS)

    cv2.imshow("Crop Images", display_image)


def trackbar_bottom(x):
    display_image = blend_image.copy()
    display_image = cv2.line(display_image, (0, 424 - x), (512, 424 - x), (255, 0, 0), LINE_THICKNESS)

    top = cv2.getTrackbarPos("Top", "Crop Images")
    display_image = cv2.line(display_image, (0, top), (512, top), (255, 0, 0), LINE_THICKNESS)

    bottom = cv2.getTrackbarPos("Bottom", "Crop Images")
    display_image = cv2.line(display_image, (0, 424 - bottom), (512, 424 - bottom), (255, 0, 0), LINE_THICKNESS)

    left = cv2.getTrackbarPos("Left", "Crop Images")
    display_image = cv2.line(display_image, (left, 0), (left, 424), (255, 0, 0), LINE_THICKNESS)

    right = cv2.getTrackbarPos("Right", "Crop Images")
    display_image = cv2.line(display_image, (512 - right, 0), (512 - right, 424), (255, 0, 0), LINE_THICKNESS)

    cv2.imshow("Crop Images", display_image)


def trackbar_left(x):
    display_image = blend_image.copy()
    display_image = cv2.line(display_image, (x, 0), (x, 424), (255, 0, 0), LINE_THICKNESS)

    top = cv2.getTrackbarPos("Top", "Crop Images")
    display_image = cv2.line(display_image, (0, top), (512, top), (255, 0, 0), LINE_THICKNESS)

    bottom = cv2.getTrackbarPos("Bottom", "Crop Images")
    display_image = cv2.line(display_image, (0, 424 - bottom), (512, 424 - bottom), (255, 0, 0), LINE_THICKNESS)

    right = cv2.getTrackbarPos("Right", "Crop Images")
    display_image = cv2.line(display_image, (512 - right, 0), (512 - right, 424), (255, 0, 0), LINE_THICKNESS)

    left = cv2.getTrackbarPos("Left", "Crop Images")
    display_image = cv2.line(display_image, (left, 0), (left, 424), (255, 0, 0), LINE_THICKNESS)

    cv2.imshow("Crop Images", display_image)


def trackbar_right(x):
    display_image = blend_image.copy()
    display_image = cv2.line(display_image, (512 - x, 0), (512 - x, 424), (255, 0, 0), LINE_THICKNESS)

    top = cv2.getTrackbarPos("Top", "Crop Images")
    display_image = cv2.line(display_image, (0, top), (512, top), (255, 0, 0), LINE_THICKNESS)

    bottom = cv2.getTrackbarPos("Bottom", "Crop Images")
    display_image = cv2.line(display_image, (0, 424 - bottom), (512, 424 - bottom), (255, 0, 0), LINE_THICKNESS)

    left = cv2.getTrackbarPos("Left", "Crop Images")
    display_image = cv2.line(display_image, (left, 0), (left, 424), (255, 0, 0), LINE_THICKNESS)

    cv2.imshow("Crop Images", display_image)


cv2.namedWindow("Crop Images")
cv2.createTrackbar("Top", "Crop Images", 0, 60, trackbar_top)
cv2.createTrackbar("Bottom", "Crop Images", 0, 60, trackbar_bottom)
cv2.createTrackbar("Left", "Crop Images", 0, 30, trackbar_left)
cv2.createTrackbar("Right", "Crop Images", 0, 30, trackbar_right)

cv2.imshow("Crop Images", blend_image)


cv2.waitKey()
print(cv2.getTrackbarPos("Top", "Crop Images"))
print(cv2.getTrackbarPos("Bottom", "Crop Images"))
print(cv2.getTrackbarPos("Left", "Crop Images"))
print(cv2.getTrackbarPos("Right", "Crop Images"))

with open("Crop_Image_data.txt", "w") as file:
    file.write("Top: " + str(cv2.getTrackbarPos("Top", "Crop Images")) + "\n")
    file.write("Bottom: " + str(cv2.getTrackbarPos("Bottom", "Crop Images")) + "\n")
    file.write("Left: " + str(cv2.getTrackbarPos("Left", "Crop Images")) + "\n")
    file.write("Right: " + str(cv2.getTrackbarPos("Right", "Crop Images")) + "\n")


cv2.destroyAllWindows()


