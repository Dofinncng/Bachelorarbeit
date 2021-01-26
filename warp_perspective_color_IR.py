import numpy as np
import cv2


IR_Image = cv2.imread("Image_Aligment/Infrared.png")

display_IR_Image = cv2.imread("Image_Aligment/Infrared.png")

color_image = cv2.imread("Image_Aligment/color.png")
display_color_image = cv2.imread("Image_Aligment/color.png")

IR_points = list()
color_points = list()

while True:
    cv2.imshow("display_IR_image", display_IR_Image)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(display_IR_Image, (x, y), 8, (0, 200, 0), 2)
            cv2.line(display_IR_Image, (x-50, y), (x+50, y), (0, 200, 0), 1)
            cv2.line(display_IR_Image, (x, y-50), (x, y+50), (0, 200, 0), 1)
            IR_points.append((x,y))

    cv2.setMouseCallback("display_IR_image", click_event)
    key = cv2.waitKey(1)

    if key == 27:
        cv2.imwrite("Image_Aligment/Perspective/infrared.png", display_IR_Image)
        cv2.destroyAllWindows()
        break

IR_points = np.float32(IR_points)


while True:
    cv2.imshow("display_color_image", display_color_image)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(display_color_image, (x, y), 15, (0, 200, 0), 6)
            cv2.line(display_color_image, (x - 100, y), (x + 100, y), (0, 200, 0), 3)
            cv2.line(display_color_image, (x, y - 100), (x, y + 100), (0, 200, 0), 3)
            color_points.append((x,y))

    cv2.setMouseCallback("display_color_image", click_event)
    key = cv2.waitKey(1)
    if key == 27:
        cv2.imwrite("Image_Aligment/Perspective/Color.png", display_color_image)
        cv2.destroyAllWindows()
        break

color_points = np.float32(color_points)

transform_mtx = cv2.getPerspectiveTransform(color_points, IR_points)
with open("Transformmatrix_color_to_infrared", "w") as file:
    file.write("transform_mtx\n")
    np.savetxt(file, transform_mtx, delimiter=",")

print(transform_mtx)

corrected_display_color_image = cv2.warpPerspective(display_color_image, transform_mtx, (512, 424))

corrected_color_image = cv2.warpPerspective(color_image, transform_mtx, (512, 424))

display_images_together = np.concatenate((corrected_display_color_image, display_IR_Image), axis=1)
images_together = np.concatenate((corrected_color_image, IR_Image), axis=1)
together = np.concatenate((display_images_together, images_together), axis=0)

cv2.imwrite("Image_Aligment/Perspective/color_new.png", corrected_color_image)
cv2.imwrite("Image_Aligment/Perspective/IR_original.png", IR_Image)
cv2.imwrite("Image_Aligment/Perspective/corrected_images.png", images_together)
cv2.imwrite("Image_Aligment/Perspective/everything.png", together)

cv2.imshow("Neues Bild", together)
cv2.waitKey()




