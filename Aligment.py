import cv2


skip = False

if skip:
    color_image = cv2.imread("Image_Aligment/Color.png")
    new_size = (int(color_image.shape[1]/2), int(color_image.shape[0]/2))
    color_image = cv2.resize(color_image, dsize=new_size)

    cv2.namedWindow("Edge_Color")
    cv2.createTrackbar("A", "Edge_Color", 0, 700, lambda x: None)
    cv2.createTrackbar("B", "Edge_Color", 0, 700, lambda x: None)

    while True:

        edge_color = cv2.Canny(color_image, cv2.getTrackbarPos("A", "Edge_Color"), cv2.getTrackbarPos("B", "Edge_Color"))
        cv2.imshow("Edge_Color", edge_color)
        key = cv2.waitKey(1)
        if key == 27:
            print(cv2.getTrackbarPos("A", "Edge_Color"))
            print(cv2.getTrackbarPos("B", "Edge_Color"))
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Image_Aligment/Edge_Detection/color.png", edge_color)


    ir_image = cv2.imread("Image_Aligment/Infrared.png")

    cv2.namedWindow("Edge_IR")
    cv2.createTrackbar("A", "Edge_IR", 0, 700, lambda x: None)
    cv2.createTrackbar("B", "Edge_IR", 0, 700, lambda x: None)

    while True:
        edge_ir = cv2.Canny(ir_image, cv2.getTrackbarPos("A", "Edge_IR"), cv2.getTrackbarPos("B", "Edge_IR"))
        cv2.imshow("Edge_IR", edge_ir)
        key = cv2.waitKey(1)
        if key == 27:
            print(cv2.getTrackbarPos("A", "Edge_IR"))
            print(cv2.getTrackbarPos("B", "Edge_IR"))
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Image_Aligment/Edge_Detection/infrared.png", edge_ir)

    edge_color = cv2.imread("Image_Aligment/Edge_Detection/color.png")
    line_array = list()
    while True:
        cv2.imshow("Edge_Color", edge_color)


        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                line_array.append((x, y))
                new_line_array = [(line_array[point -1], line_array[point]) for point in range(len(line_array)) if point % 2 != 0]
                print(new_line_array)
                for coordinates in new_line_array:
                    try:
                        pitch = (coordinates[1][1] - coordinates[0][1]) / (coordinates[1][0] - coordinates[0][0])
                        b = coordinates[0][1] - pitch * coordinates[0][0]
                        cv2.line(edge_color, (0, int(b)), (960, int(pitch * 960 + b)), (150, 0, 0), 2)
                        cv2.putText(edge_color,
                                    str(pitch)[:6] + "x +" + str(b)[0:3],
                                    (coordinates[0][0], coordinates[0][1]),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1)
                    except ZeroDivisionError:
                        x_value = coordinates[0][0]
                        cv2.line(edge_color, (x_value, 0), (x_value, 540), (150, 0, 0), 2)
                        cv2.putText(edge_color,
                                    "x = " + str(coordinates[0][0]),
                                    (coordinates[0][0], coordinates[0][1]),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1)

        cv2.setMouseCallback("Edge_Color", click_event)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Image_Aligment/Edge_Detection_with_lines/color.png", edge_color)

    edge_IR = cv2.imread("Image_Aligment/Edge_Detection/infrared.png")
    line_array = list()
    while True:
        cv2.imshow("Edge_IR", edge_IR)

        def click_event_ir(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
                line_array.append((x, y))
                new_line_array = [(line_array[point - 1], line_array[point]) for point in range(len(line_array)) if
                                  point % 2 != 0]
                print(new_line_array)
                for coordinates in new_line_array:
                    try:
                        pitch = (coordinates[1][1] - coordinates[0][1]) / (coordinates[1][0] - coordinates[0][0])
                        b = coordinates[0][1] - pitch * coordinates[0][0]
                        cv2.line(edge_IR, (0, int(b)), (512, int(pitch * 512 + b)), (150, 0, 0), 2)
                        cv2.putText(edge_IR,
                                    str(pitch)[:6] + "x +" + str(b)[0:3],
                                    (coordinates[0][0], coordinates[0][1]),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1)
                    except ZeroDivisionError:
                        x_value = coordinates[0][0]
                        cv2.line(edge_IR, (x_value, 0), (x_value, 424), (150, 0, 0), 2)
                        cv2.putText(edge_IR,
                                    "x = " + str(coordinates[0][0]),
                                    (coordinates[0][0], coordinates[0][1]),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1)


        cv2.setMouseCallback("Edge_IR", click_event_ir)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            cv2.imwrite("Image_Aligment/Edge_Detection_with_lines/infrared.png", edge_IR)

elif not skip:
    print("ok")




