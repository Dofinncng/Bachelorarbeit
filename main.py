import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
import PIL.Image, PIL.ImageTk
from matplotlib.ticker import MaxNLocator
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

LINE_AMOUNT = 30
kinect_color = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
kinect_depth = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)


class GraphicalUserInterface:

    def __init__(self, master):
        master.geometry("1400x700+30+30")
        master.title("Graphical User Interface")

        # Start Frame

        self.start_frame = tk.Frame(master)

        self.start_frame.pack(expand=True, fill="both")
        self.start_frame.pack_propagate(0)
        self.start_frame_label = tk.Label(self.start_frame, text="Start Frame")
        self.start_frame_label.pack()

        self.continue_button_start_frame = tk.Button(self.start_frame, text="Take Image and Continue",
                                                     command=lambda: self.start_to_image_frame())
        self.continue_button_start_frame.place(relx=0.85, rely=0.05)

        self.color_stream_label = tk.Label(self.start_frame)
        self.color_stream_label.place(relx=0.6, rely=0.2)
        self.color_stream_label.after(100, self.color_stream)

        self.depth_stream_label = tk.Label(self.start_frame, bg="Yellow")
        self.depth_stream_label.place(relx=0.05, rely=0.2)
        self.depth_stream_label.after(100, self.depth_stream)

        self.width, self.height = 480, 270

        # Image Frame

        self.image_frame = tk.Frame(master)
        self.image_frame_label = tk.Label(self.image_frame, text="Image Frame")
        self.image_frame_label.pack()

        self.continue_button_image_frame = tk.Button(self.image_frame, text="Continue",
                                                     command=lambda:
                                                     self.image_to_precise_frame())
        self.continue_button_image_frame.place(relx=0.85, rely=0.05)

        self.back_button_image_frame = tk.Button(self.image_frame, text="Back",
                                                     command=lambda: self.image_to_start_frame())
        self.back_button_image_frame.place(relx=0.15, rely=0.05)

        self.image_panel = tk.Label(self.image_frame)
        self.image_panel.place(x=750, y=550, anchor="sw")

        self.arrow_x_canvas = tk.Canvas(self.image_frame)
        self.arrow_x_canvas.config(width=210, height=30)
        self.arrow_x_canvas.place(x=750, y=565)
        self.arrow_x_canvas.create_line(0, 15, 200, 15, arrow=tk.LAST)
        self.x_axis_tag = tk.Label(self.image_frame, text="x")
        self.x_axis_tag.place(x=960, y=590, anchor="s")

        self.arrow_y_canvas = tk.Canvas(self.image_frame)
        self.arrow_y_canvas.config(width=30, height=210)
        self.arrow_y_canvas.place(x=735, y=550, anchor="se")
        self.arrow_y_canvas.create_line(15, 210, 15, 10, arrow=tk.LAST)
        self.y_axis_tag = tk.Label(self.image_frame, text="y")
        self.y_axis_tag.place(x=720, y=320)

        self.x_entry_widget = tk.Entry(self.image_frame, bg="yellow", text="10")
        self.x_entry_widget.place(x=900, y=610)
        self.x_entry_tag = tk.Label(self.image_frame, text="x")
        self.x_entry_tag.place(x=900, y=610, anchor="ne")

        self.y_entry_widget = tk.Entry(self.image_frame, bg="yellow")
        self.y_entry_widget.place(x=900, y=630)
        self.y_entry_tag = tk.Label(self.image_frame, text="y")
        self.y_entry_tag.place(x=900, y=630, anchor="ne")

        ########################################################

        #self.vertical_display = tk.Label(self.image_frame, text=30, bg="red")
        #self.vertical_display.place(relx=0.1, rely=0.9)

        #self.horizontal_display = tk.Label(self.image_frame, text=30, bg="red")
        #self.horizontal_display.place(relx=0.1, rely=0.85)

        #self.up_button = tk.Button(master=self.image_frame, text="hoch",
        #                      command=lambda: self.turn_horizontal(self.horizontal_display, 5))
        #self.up_button.place(x=300, y=570)

        #self.down_button = tk.Button(master=self.image_frame, text="runter",
        #                        command=lambda: self.turn_horizontal(self.horizontal_display, -5))
        #self.down_button.place(x=300, y=620)

        #self.left_button = tk.Button(master=self.image_frame, text="links",
        #                        command=lambda: self.turn_vertical(self.vertical_display, -5))
        #self.left_button.place(x=250, y=595)

        #self.right_button = tk.Button(master=self.image_frame, text="rechts",
        #                         command=lambda: self.turn_vertical(self.vertical_display, 5))
        #self.right_button.place(x=350, y=595)

        # Precise Frame

        self.precise_frame = tk.Frame(master)
        self.precise_frame_label = tk.Label(self.precise_frame, text="Precise Frame")
        self.precise_frame_label.pack()

        self.x_value_label = tk.Label(self.precise_frame, bg="red")
        self.x_value_label.pack()

        self.y_value_label = tk.Label(self.precise_frame, bg="red")
        self.y_value_label.pack()

        self.image_panel_precise_frame = tk.Label(self.precise_frame)
        self.image_panel_precise_frame.place(x=750, y= 550, anchor="sw")

        self.back_button_precise_frame = tk.Button(self.precise_frame, text="Back",
                                                   command=lambda: self.precise_to_image_frame())
        self.back_button_precise_frame.place(relx=0.15, rely=0.05)

    def color_stream(self):
        color_frame = kinect_color.get_last_color_frame()

        color_frame = np.reshape(color_frame, (2073600, 4))
        color_frame = color_frame[:, 0:3]

        # extract then combine the RBG data
        colour_frame_red = color_frame[:, 0]
        colour_frame_red = np.reshape(colour_frame_red, (1080, 1920))
        color_frame_green = color_frame[:, 1]
        color_frame_green = np.reshape(color_frame_green, (1080, 1920))
        color_frame_blue = color_frame[:, 2]
        color_frame_blue = np.reshape(color_frame_blue, (1080, 1920))
        self.full_color_frame = cv2.merge([colour_frame_red, color_frame_green, color_frame_blue])
        self.full_color_frame = self.full_color_frame[0:1080, 308:1800]
        self.full_color_frame = cv2.resize(self.full_color_frame, (512, 424))
        self.full_color_frame = cv2.cvtColor(self.full_color_frame, cv2.COLOR_BGR2RGB)


        self.color_image = PIL.Image.fromarray(self.full_color_frame)
        self.color_image_tk = PIL.ImageTk.PhotoImage(image=self.color_image)
        self.color_stream_label.image_tk = self.color_image_tk
        self.color_stream_label["image"] = self.color_image_tk
        self.color_stream_label.after(100, self.color_stream)

    def depth_stream(self):
        depth_frame = kinect_depth.get_last_depth_frame()
        depth_frame = depth_frame.astype(np.uint8)
        depth_frame = np.reshape(depth_frame, (424, 512))
        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)

        self.depth_image = PIL.Image.fromarray(depth_frame)
        self.depth_image_tk = PIL.ImageTk.PhotoImage(image=self.depth_image)
        self.depth_stream_label.image_tk = self.depth_image_tk
        self.depth_stream_label["image"] = self.depth_image_tk
        self.depth_stream_label.after(100, self.depth_stream)

    def get_depth_data(self):
        depth_frame = kinect_depth.get_last_depth_frame()
        depth_frame = depth_frame.astype(np.uint8)
        depth_frame = np.reshape(depth_frame, (424, 512))
        return depth_frame

    def get_fig(self, data):

        plot_data = list()
        for y in range(len(data)):
            for x in range(len(data[y])):
                if 140 < data[y, x] < 205 and 100 < y < 300 and 150 < x < 400:
                    plot_data.append(np.array([x, 424 - y, 255 - data[y, x]]))
                else:
                    plot_data.append(np.array([x, 424 - y, 50]))

        self.plot_data = np.asarray(plot_data)

        #print(plot_data)

        #self.plot_data = list()

        #for x in range(int(ARRAY_WIDTH / plot_square_size)):
        #    x = x * plot_square_size
        #    for y in range(int(ARRAY_HEIGHT / plot_square_size)):
        #        y = y * plot_square_size
        #        single_plot_item = np.array([x + plot_square_size / 2 - 0.5,
        #                                     y + plot_square_size / 2 - 0.5,
        #                                     sensor_input[y:y + plot_square_size, x:x + plot_square_size].mean()])
        #        self.plot_data.append(single_plot_item)
        #
        # eventuell noch alle Randpunkte einbeziehen
        #
        #self.plot_data = np.asarray(self.plot_data)

        x = self.plot_data[:, 0]
        y = self.plot_data[:, 1]
        z = self.plot_data[:, 2]

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlim(50, 300)
        plt.xlabel("X Achse")
        plt.ylabel("Y Achse")

        surf = ax.plot_trisurf(x, y, z, cmap="plasma", linewidth=0)
        fig.colorbar(surf)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(5))

        fig.tight_layout()

        #ax.view_init(90, -90)
        return fig

    def get_precise_fig(self, data, x_value, y_value):

        first_point = (x_value / LINE_AMOUNT * 512, y_value / LINE_AMOUNT * 424)
        second_point = ((x_value + 1) / LINE_AMOUNT * 512, (y_value + 1) / LINE_AMOUNT * 424)

        print(first_point)
        print(second_point)

        selected_data = list()
        diselected_data = list()
        for row in data:
            if first_point[0] <= row[0] <= second_point[0] and first_point[1] <= row[1] <= second_point[1]:
                selected_data.append(row)
            else:
                diselected_data.append(row)

        selected_data = np.asarray(selected_data)
        diselected_data = np.asarray(diselected_data)

        print(selected_data)
        print(diselected_data)

        x_diselected = diselected_data[:, 0]
        y_diselected = diselected_data[:, 1]
        z_diselected = diselected_data[:, 2]

        x_selected = selected_data[:, 0]
        y_selected = selected_data[:, 1]
        z_selected = selected_data[:, 2]

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlim(5, 300)

        surf = ax.plot_trisurf(x_diselected, y_diselected, z_diselected, cmap="plasma", linewidth=0, alpha=0.3)
        fig.colorbar(surf)

        ax.plot_trisurf(x_selected, y_selected, z_selected, color="red", linewidth=0)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(5))

        fig.tight_layout()

        X_plane = np.array([[first_point[0], second_point[0]], [first_point[0], second_point[0]]])
        Y_plane = np.array([[first_point[1], first_point[1]], [second_point[1], second_point[1]]])
        Z_plane = np.array([[210, 210], [210, 210]])

        plt.xlabel("X Achse")
        plt.ylabel("Y Achse")

        ax.plot_surface(X_plane, Y_plane, Z_plane, color="green")

        #ax.view_init(90, -90)
        return fig



    #def get_dummy_three_d_image(self):
    #    ARRAY_HEIGHT = 270
    #    ARRAY_WIDTH = 480

    #    sensor_input = np.random.rand(ARRAY_HEIGHT, ARRAY_WIDTH)

    #    plot_square_size = 6

    #    if ARRAY_HEIGHT % plot_square_size or ARRAY_WIDTH % plot_square_size:
    #        print("Fehlermeldung")

    #    self.plot_data = list()

    #    for x in range(int(ARRAY_WIDTH / plot_square_size)):
    #        x = x * plot_square_size
    #        for y in range(int(ARRAY_HEIGHT / plot_square_size)):
    #            y = y * plot_square_size
    #            single_plot_item = np.array([x + plot_square_size / 2 - 0.5,
    #                                         y + plot_square_size / 2 - 0.5,
    #                                         sensor_input[y:y + plot_square_size, x:x + plot_square_size].mean()])
    #            self.plot_data.append(single_plot_item)
        #
        # eventuell noch alle Randpunkte einbeziehen
        #
    #    self.plot_data = np.asarray(self.plot_data)

    #    x = self.plot_data[:, 0]
    #    y = self.plot_data[:, 1]
    #    z = self.plot_data[:, 2]

    #    fig = plt.figure()

    #    ax = fig.add_subplot(111, projection="3d")
    #    ax.set_zlim(0, 5)

    #    surf = ax.plot_trisurf(x, y, z, cmap="plasma", linewidth=0)
    #    fig.colorbar(surf)

    #    ax.xaxis.set_major_locator(MaxNLocator(5))
    #    ax.yaxis.set_major_locator(MaxNLocator(5))
    #    ax.zaxis.set_major_locator(MaxNLocator(5))

    #    fig.tight_layout()

        #ax.view_init(turn_vertical, turn_horizontal)
    #    return fig

    #def turn_vertical(self, display, value):
    #    display["text"] = display["text"] + value

    #def turn_horizontal(self, display, value):
    #    display["text"] = display["text"] + value

    def start_to_image_frame(self):
        self.start_frame.pack_forget()

        self.fixed_color_image = cv2.cvtColor(self.full_color_frame, cv2.COLOR_BGR2RGB)

        cluster_color_image = self.draw_cluster_on_image()
        self.image_panel.cluster_color_image = cluster_color_image
        self.image_panel["image"] = cluster_color_image

        self.depth_data = self.get_depth_data()
        self.fig = self.get_fig(self.depth_data)
        #fig = self.get_dummy_three_d_image()
        plot_canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        plot_canvas._tkcanvas.place(x=10, y=550, anchor="sw")

        self.image_frame.pack(expand=True, fill="both")
        self.image_frame.pack_propagate(0)

    def image_to_precise_frame(self):
        self.image_frame.pack_forget()

        self.highlighted_image = self.highlight_location_on_image()

        self.precise_fig = self.get_precise_fig(self.plot_data,
                                                int(self.x_entry_widget.get()),
                                                int(self.y_entry_widget.get()))

        self.precise_plot_canvas = FigureCanvasTkAgg(self.precise_fig, master=self.precise_frame)
        self.precise_plot_canvas._tkcanvas.place(x=10, y=550, anchor="sw")

        self.x_value_label["text"] = self.x_entry_widget.get()
        self.y_value_label["text"] = self.y_entry_widget.get()

        self.image_panel_precise_frame["image"] = self.highlighted_image
        self.precise_frame.pack(expand=True, fill="both")
        self.precise_frame.pack_propagate(0)

    def image_to_start_frame(self):
        self.image_frame.pack_forget()
        self.start_frame.pack(expand=True, fill="both")
        self.start_frame.pack_propagate(0)

    def precise_to_image_frame(self):
        self.precise_frame.pack_forget()
        self.image_frame.pack(expand=True, fill="both")
        self.image_frame.pack_propagate(0)

    def draw_cluster_on_image(self):
        image = self.fixed_color_image
        #image = self.fixed_color_image

        #image = cv2.flip(image, 1)

        height, width, channel = image.shape

        line_distance_x = width / LINE_AMOUNT
        line_distance_y = height / LINE_AMOUNT

        for line in range(LINE_AMOUNT):
            image = cv2.line(image, (int(line * line_distance_x), 0), (int(line * line_distance_x), height),
                             (100, 100, 100))
            image = cv2.line(image, (0, int(line * line_distance_y)), (width, int(line * line_distance_y)),
                             (100, 100, 100))

        image_as_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image_as_array)
        image = PIL.ImageTk.PhotoImage(image)
        return image

    def highlight_location_on_image(self):
        image = self.fixed_color_image
        #image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape

        first_point = int((int(self.x_entry_widget.get())) / LINE_AMOUNT * width), \
                      int(height - (int(self.y_entry_widget.get()) + 1) / LINE_AMOUNT * height)

        second_point = int((int(self.x_entry_widget.get()) + 1) / LINE_AMOUNT * width), \
                       int(height - (int(self.y_entry_widget.get())) / LINE_AMOUNT * height)

        blurred_image = cv2.GaussianBlur(image, (29, 29), 0)
        mask = np.zeros((image.shape), dtype=np.uint8)
        mask = cv2.rectangle(mask, first_point, second_point, (255, 255, 255), -1)

        image = np.where(mask == np.array([255, 255, 255]), image, blurred_image)

        image = PIL.Image.fromarray(image)
        image = PIL.ImageTk.PhotoImage(image)
        return image

    #def get_dummy_precise_three_d_image(self, turn_vertical, turn_horizontal, plot_data, x_value, y_value):

    #    x_value, y_value = int(x_value), int(y_value)
    #    first_point = (x_value/LINE_AMOUNT*self.width, y_value/LINE_AMOUNT*self.height)
    #    second_point = ((x_value+1)/LINE_AMOUNT*self.width, (y_value+1)/LINE_AMOUNT*self.height)
    #    selected_data = list()
    #    diselected_data = list()
    #    for row in plot_data:
    #        if first_point[0] <= row[0] <= second_point[0] and first_point[1] <= row[1] <= second_point[1]:
    #            selected_data.append(row)
    #        else:
    #            diselected_data.append(row)

    #    selected_data = np.asarray(selected_data)
    #    diselected_data = np.asarray(diselected_data)

    #    x_diselected = diselected_data[:, 0]
    #    y_diselected = diselected_data[:, 1]
    #    z_diselected = diselected_data[:, 2]

    #    x_selected = selected_data[:, 0]
    #    y_selected = selected_data[:, 1]
    #    z_selected = selected_data[:, 2]

    #    fig = plt.figure()

    #    ax = fig.add_subplot(111, projection="3d")
    #    ax.set_zlim(0, 5)

    #    surf = ax.plot_trisurf(x_diselected, y_diselected, z_diselected, cmap="plasma", linewidth=0, alpha=0.3)
    #    fig.colorbar(surf)

    #    ax.plot_trisurf(x_selected, y_selected, z_selected, color="red", linewidth=0)

    #    ax.xaxis.set_major_locator(MaxNLocator(5))
    #    ax.yaxis.set_major_locator(MaxNLocator(5))
    #    ax.zaxis.set_major_locator(MaxNLocator(5))

    #    fig.tight_layout()


    #    X_plane = np.array([[first_point[0], second_point[0]], [first_point[0], second_point[0]]])
    #    Y_plane = np.array([[first_point[1], first_point[1]], [second_point[1], second_point[1]]])
    #    Z_plane = np.array([[4.5, 4.5], [4.5, 4.5]])

    #    ax.plot_surface(X_plane, Y_plane, Z_plane, color="green")

    #    ax.view_init(turn_vertical, turn_horizontal)
    #    return fig

root = tk.Tk()
GraphicalUserInterface(root)
root.mainloop()