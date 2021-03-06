import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import numpy as np
import PIL.Image, PIL.ImageTk
from matplotlib.ticker import MaxNLocator
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import math

LINE_AMOUNT = 130
PLANE_DISTANCE = 100
PLANE_ERROR_LIMIT = 90000
PLANE_ERROR_AMOUNT = 10
kinect_color = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
kinect_depth = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
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

image_height, image_width = 424 - crop_data[0] - crop_data[1], 512 - crop_data[2] - crop_data[3]


class GraphicalUserInterface:

    """Class for Graphical User Interface. Uses Microsoft xBox Kinect Sensor
    for acquiring Color and Depth Images.
    Lays cluster above Image to select a part of the image
    which is then driven to by robot"""

    def __init__(self, master):
        master.geometry("1300x600+30+30")
        master.title("Graphical User Interface")
        root.protocol("WM_DELETE_WINDOW", lambda: root.quit())

        # Start Frame
        self.start_frame = tk.Frame(master)
        self.start_frame.pack(expand=True, fill="both")
        self.start_frame.pack_propagate(0)
        self.start_frame_label = tk.Label(self.start_frame, text="Start Frame")
        self.start_frame_label.pack()
        self.continue_button_start_frame = tk.Button(self.start_frame,
                                                     text="Take Image and Continue",
                                                     command=lambda:
                                                     self.start_to_image_frame())
        self.continue_button_start_frame.place(relx=0.85, rely=0.05)
        self.start_frame_shortcut_state = tk.BooleanVar()
        self.start_frame_shortcut_state.set(False)
        self.start_frame_shortcut_box = tk.Checkbutton(self.start_frame,
                                                       text="take shortcut",
                                                       var=self.start_frame_shortcut_state)
        self.start_frame_shortcut_box.place(relx=0.75, rely=0.05)
        self.color_stream_label = tk.Label(self.start_frame)
        self.color_stream_label.place(relx=0.6, rely=0.2)
        self.color_stream_label.after(100, self.color_stream)
        self.depth_stream_label = tk.Label(self.start_frame)
        self.depth_stream_label.place(relx=0.05, rely=0.2)
        self.depth_stream_label.after(100, self.depth_stream)
        # Image Frame
        self.image_frame = tk.Frame(master)
        self.image_frame_label = tk.Label(self.image_frame,
                                          text="Image Frame")
        self.image_frame_label.pack()
        self.continue_button_image_frame = tk.Button(self.image_frame,
                                                     text="Continue",
                                                     command=lambda:
                                                     self.image_to_precise_frame())
        self.continue_button_image_frame.place(relx=0.85, rely=0.05)
        self.back_button_image_frame = tk.Button(self.image_frame,
                                                 text="Back",
                                                 command=lambda: self.image_to_start_frame())
        self.back_button_image_frame.place(relx=0.15, rely=0.05)
        self.image_frame_shortcut_state = tk.BooleanVar()
        self.image_frame_shortcut_state.set(False)
        self.image_frame_shortcut_box = tk.Checkbutton(self.image_frame,
                                                       text="take shortcut",
                                                       var=self.image_frame_shortcut_state)
        self.image_frame_shortcut_box.place(relx=0.75, rely=0.05)
        self.image_panel = tk.Canvas(self.image_frame,
                                     width=image_width,
                                     height=image_height)
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
        self.x_entry_widget = tk.Entry(self.image_frame, bg="yellow")
        self.x_entry_widget.place(x=900, y=100)
        self.x_entry_tag = tk.Label(self.image_frame, text="x")
        self.x_entry_tag.place(x=900, y=100, anchor="ne")
        self.y_entry_widget = tk.Entry(self.image_frame, bg="yellow")
        self.y_entry_widget.place(x=900, y=130)
        self.y_entry_tag = tk.Label(self.image_frame, text="y")
        self.y_entry_tag.place(x=900, y=130, anchor="ne")
        # Precise Frame
        self.precise_frame = tk.Frame(master)
        self.precise_frame_label = tk.Label(self.precise_frame, text="Precise Frame")
        self.precise_frame_label.pack()
        self.x_value_label = tk.Label(self.precise_frame, bg="red")
        self.x_value_label.pack()
        self.y_value_label = tk.Label(self.precise_frame, bg="red")
        self.y_value_label.pack()
        self.image_panel_precise_frame = tk.Label(self.precise_frame)
        self.image_panel_precise_frame.place(x=750, y=550, anchor="sw")
        self.back_button_precise_frame = tk.Button(self.precise_frame,
                                                   text="Back",
                                                   command=lambda:
                                                   self.precise_to_image_frame())
        self.back_button_precise_frame.place(relx=0.15, rely=0.05)
        self.center_point_widget = tk.Label(self.precise_frame,
                                            bg="red",
                                            text="Mittelpunkt")
        self.center_point_widget.place(x=900, y=100, anchor="nw")
        self.plane_equation_widget = tk.Label(self.precise_frame,
                                              bg="red",
                                              text="Ebenengleichung")
        self.plane_equation_widget.place(x=900, y=130, anchor="nw")
        self.center_point_label = tk.Label(self.precise_frame,
                                           text="Center Point")
        self.center_point_label.place(x=900, y=100, anchor="ne")
        self.plane_equation_label = tk.Label(self.precise_frame,
                                             text="Plane Equation")
        self.plane_equation_label.place(x=900, y=130, anchor="ne")

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
        full_color_frame = cv2.merge([colour_frame_red,
                                      color_frame_green,
                                      color_frame_blue])
        full_color_frame = cv2.cvtColor(full_color_frame,
                                        cv2.COLOR_BGR2RGB)
        full_color_frame = cv2.undistort(full_color_frame,
                                         mtx_color,
                                         dist_color,
                                         None,
                                         newcameramtx_color)
        self.cv2_color_image = cv2.warpPerspective(full_color_frame,
                                                   transform_mtx,
                                                   (512, 424))
        self.cv2_color_image = self.cv2_color_image[crop_data[0]: 424 - crop_data[1],
                               crop_data[2]: 512 - crop_data[3]]
        self.color_image = PIL.Image.fromarray(self.cv2_color_image)
        self.color_image_tk = PIL.ImageTk.PhotoImage(image=self.color_image)
        self.color_stream_label.image_tk = self.color_image_tk
        self.color_stream_label["image"] = self.color_image_tk
        self.color_stream_label.after(100, self.color_stream)

    def get_color_image(self):
        color_frame = kinect_color.get_last_color_frame()
        color_frame = np.reshape(color_frame, (2073600, 4))
        color_frame = color_frame[:, 0:3]
        # extract then combine the RBG data
        colour_frame_red = color_frame[:, 0]
        colour_frame_red = np.reshape(colour_frame_red,
                                      (1080, 1920))
        color_frame_green = color_frame[:, 1]
        color_frame_green = np.reshape(color_frame_green,
                                       (1080, 1920))
        color_frame_blue = color_frame[:, 2]
        color_frame_blue = np.reshape(color_frame_blue,
                                      (1080, 1920))
        full_color_frame = cv2.merge([colour_frame_red,
                                      color_frame_green,
                                      color_frame_blue])
        full_color_frame = cv2.cvtColor(full_color_frame,
                                        cv2.COLOR_BGR2RGB)
        full_color_frame = cv2.undistort(full_color_frame,
                                         mtx_color,
                                         dist_color,
                                         None,
                                         newcameramtx_color)
        self.cv2_color_image = cv2.warpPerspective(full_color_frame,
                                                   transform_mtx,
                                                   (512, 424))
        self.cv2_color_image = self.cv2_color_image[crop_data[0]: 424 - crop_data[1],
                               crop_data[2]: 512 - crop_data[3]]
        self.color_image = PIL.Image.fromarray(self.cv2_color_image)
        color_image_tk = PIL.ImageTk.PhotoImage(image=self.color_image)
        return color_image_tk

    def depth_stream(self):
        depth_frame = kinect_depth.get_last_depth_frame()
        depth_frame = depth_frame.astype(np.uint8)
        depth_frame = np.reshape(depth_frame, (424, 512))
        depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

        depth_frame = cv2.undistort(depth_frame,
                                    mtx_IR,
                                    dist_IR,
                                    None,
                                    newcameramtx_IR)
        self.depth_frame = depth_frame[crop_data[0]: 424 - crop_data[1],
                           crop_data[2]: 512 - crop_data[3]]
        self.depth_image = PIL.Image.fromarray(self.depth_frame)
        self.depth_image_tk = PIL.ImageTk.PhotoImage(image=self.depth_image)
        self.depth_stream_label.image_tk = self.depth_image_tk
        self.depth_stream_label["image"] = self.depth_image_tk
        self.depth_stream_label.after(100, self.depth_stream)

    def get_depth_data(self):
        depth_frame = kinect_depth.get_last_depth_frame()
        depth_frame = depth_frame.astype(np.uint8)
        depth_frame = np.reshape(depth_frame, (424, 512))
        undistored_depth_frame = cv2.undistort(depth_frame,
                                               mtx_IR,
                                               dist_IR,
                                               None,
                                               newcameramtx_IR)
        undistored_depth_frame = undistored_depth_frame[crop_data[0]: 424 - crop_data[1],
                                 crop_data[2]: 512 - crop_data[3]]
        return undistored_depth_frame

    def get_fig(self, data):
        plot_data = list()
        limit = 159
        data[data > limit] = limit
        for y in range(len(data)):
            for x in range(len(data[y])):
                plot_data.append(np.array([x, image_height - y, 255 - data[y, x] - 96]))
        self.plot_data = np.asarray(plot_data)
        x = self.plot_data[:, 0]
        y = self.plot_data[:, 1]
        z = self.plot_data[:, 2]
        fig = plt.figure()
        if not self.start_frame_shortcut_state.get():
            ax = fig.add_subplot(111, projection="3d")
            plt.xlabel("X Achse")
            plt.ylabel("Y Achse")
            surf = ax.plot_trisurf(x, y, z, cmap="plasma", linewidth=0)
            fig.colorbar(surf)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.zaxis.set_major_locator(MaxNLocator(5))
        elif self.start_frame_shortcut_state.get():
            ax = plt.axes(projection='3d')
            ax.scatter(x, y, z, c=z, cmap='plasma', linewidth=0.5, s=0.1)
        ax.set_zlim(0, 486)
        fig.tight_layout()
        return fig

    def calculate_plane_equation(self, data):
        xs = data[:, 0]
        ys = data[:, 1]
        zs = data[:, 2]
        tmp_A = []
        tmp_b = []
        for i in range(len(xs)):
            tmp_A.append(np.array([xs[i], ys[i], 1]))
            tmp_b.append(np.array([zs[i]]))
        b = np.asarray(tmp_b)
        c = np.asarray(tmp_A)
        c_transpose = c.T
        # Manual solution
        fit = np.dot(np.dot(np.linalg.inv(np.dot(c_transpose, c)), c_transpose), b)
        errors = np.dot((b - c), fit)
        residual = np.linalg.norm(errors)
        new_fit = fit[0], fit[1], fit[2] + PLANE_DISTANCE
        gap = PLANE_DISTANCE / math.sqrt(fit[0]**2 + fit[1]**2 + 1)
        x_limit = (data[0][0] - fit[0] * gap)[0], (data[0][1] - fit[1] * gap)[0]
        y_limit = (data[-1][0] - fit[0] * gap)[0], (data[-1][1] - fit[1] * gap)[0]
        # calculate center
        x_center = x_limit[0] + (y_limit[0] - x_limit[0])/2
        y_center = y_limit[1] + (x_limit[1] - y_limit[1])/2
        z_center = fit[0]*x_center + fit[1]*y_center + fit[2] + PLANE_DISTANCE
        center = x_center, y_center, z_center
        return new_fit, x_limit, y_limit, errors, center

    def get_precise_fig(self, data, x_value, y_value):
        first_point = (x_value / LINE_AMOUNT * image_width,
                       y_value / LINE_AMOUNT * image_height)
        second_point = ((x_value + 1) / LINE_AMOUNT * image_width,
                        (y_value + 1) / LINE_AMOUNT * image_height)
        selected_data = list()
        disselected_data = list()
        for row in data:
            if first_point[0] <= row[0] <= second_point[0] \
                    and first_point[1] <= row[1] <= second_point[1]:
                selected_data.append(row)
            else:
                disselected_data.append(row)
        selected_data = np.asarray(selected_data)
        disselected_data = np.asarray(disselected_data)
        x_diselected = disselected_data[:, 0]
        y_diselected = disselected_data[:, 1]
        z_diselected = disselected_data[:, 2]
        x_selected = selected_data[:, 0]
        y_selected = selected_data[:, 1]
        z_selected = selected_data[:, 2]
        fig = plt.figure()
        if not self.image_frame_shortcut_state.get():
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_trisurf(x_diselected,
                                   y_diselected,
                                   z_diselected,
                                   cmap="plasma",
                                   linewidth=0,
                                   alpha=0.3)
            fig.colorbar(surf)
            ax.plot_trisurf(x_selected,
                            y_selected,
                            z_selected,
                            color="red",
                            linewidth=0)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.zaxis.set_major_locator(MaxNLocator(5))
        elif self.image_frame_shortcut_state.get():
            ax = plt.axes(projection='3d')
            ax.scatter(x_selected,
                       y_selected,
                       z_selected,
                       c="red",
                       linewidth=0.5,
                       s=0.1)
            ax.scatter(x_diselected, y_diselected, z_diselected, c=z_diselected,
                       cmap="plasma", linewidth=0.5, s=0.1, alpha=0.3)
        fig.tight_layout()
        fit, x_limit, y_limit, errors, center = self.calculate_plane_equation(selected_data)
        X, Y = np.meshgrid(np.arange(x_limit[0], y_limit[0]),
                           np.arange(y_limit[1], x_limit[1]))
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r, c] = fit[0] * X[r, c] + fit[1] * Y[r, c] + fit[2]
        ax.plot_surface(X, Y, Z, color='green')
        ax.scatter(center[0], center[1], center[2], c="red", s=10)
        ax.set_zlim(0, 486)
        plt.xlabel("X Achse")
        plt.ylabel("Y Achse")
        return fig, fit, errors, center

    def callback(self, event):
        step_x = image_width/LINE_AMOUNT
        step_y = image_height/LINE_AMOUNT
        for i in range(LINE_AMOUNT):
            if i * step_x < event.x <= i * step_x + step_x:
                x_value = i
            if i * step_y < event.y <= i * step_y + step_y:
                y_value = i
        self.x_entry_widget.delete(0, tk.END)
        self.x_entry_widget.insert(0, x_value)
        self.y_entry_widget.delete(0, tk.END)
        self.y_entry_widget.insert(0, LINE_AMOUNT - 1 - y_value)

    def start_to_image_frame(self):
        if self.start_frame_shortcut_state.get():
            self.image_frame_shortcut_state.set(True)
        self.start_frame.pack_forget()
        self.image_frame_color_image = self.get_color_image()
        self.image_panel.create_image(0,
                                      0,
                                      image=self.image_frame_color_image,
                                      anchor=tk.NW)
        self.image_panel.bind("<Button-1>", self.callback)
        line_distance_x = image_width / LINE_AMOUNT
        line_distance_y = image_height / LINE_AMOUNT
        for line in range(LINE_AMOUNT):
            self.image_panel.create_line(line * line_distance_x,
                                         0,
                                         line * line_distance_x,
                                         image_height,
                                         width=0.01, fill="yellow")
            self.image_panel.create_line(0,
                                         line * line_distance_y,
                                         image_width,
                                         line * line_distance_y,
                                         width=0.01, fill="yellow")
        self.depth_data = self.get_depth_data()
        self.fig = self.get_fig(self.depth_data)
        plot_canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        plot_canvas._tkcanvas.place(x=10, y=550, anchor="sw")
        self.image_frame.pack(expand=True, fill="both")
        self.image_frame.pack_propagate(0)

    def image_to_precise_frame(self):
        if not self.x_entry_widget.get().isnumeric():
            messagebox.showerror("Fehler",
                                 "Bitte numerischen ganzzahligen Wert eingeben")
        elif not self.y_entry_widget.get().isnumeric():
            messagebox.showerror("Fehler",
                                 "Bitte numerischen ganzzahligen Wert eingeben")
        elif not 0 <= int(self.x_entry_widget.get()) <= LINE_AMOUNT - 1:
            messagebox.showerror("Fehler",
                                 " Bitte Wert zwischen 0 und " +
                                 str(LINE_AMOUNT-1) +
                                 " eingeben.")
        elif not 0 <= int(self.y_entry_widget.get()) <= LINE_AMOUNT - 1:
            messagebox.showerror("Fehler",
                                 " Bitte Wert zwischen 0 und " +
                                 str(LINE_AMOUNT - 1) +
                                 " eingeben.")
        else:
            self.precise_fig, \
            self.fit, errors, \
            center = self.get_precise_fig(self.plot_data,
                                          int(self.x_entry_widget.get()),
                                          int(self.y_entry_widget.get()))
            error_amount = 0
            for error in errors:
                if abs(error[0]) > PLANE_ERROR_LIMIT:
                    error_amount = error_amount + 1
            if error_amount > PLANE_ERROR_AMOUNT:
                messagebox.showerror("Fehler",
                                     "Fuer diesen Bildbereich konnte"
                                     "keine vailde Ebene geunden werden")
            else:
                self.image_frame.pack_forget()
                self.highlighted_image = self.highlight_location_on_image()
                self.precise_plot_canvas = FigureCanvasTkAgg(self.precise_fig,
                                                             master=self.precise_frame)
                self.precise_plot_canvas._tkcanvas.place(x=10, y=550, anchor="sw")
                self.x_value_label["text"] = self.x_entry_widget.get()
                self.y_value_label["text"] = self.y_entry_widget.get()
                self.plane_equation_widget["text"] = "%f x + %f y + %f = z" %(self.fit[0],
                                                                              self.fit[1],
                                                                              self.fit[2])
                self.center_point_widget["text"] = "(x = %f, y = %f, z= %f)" % (center[0],
                                                                                center[1],
                                                                                center[2])
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

    def highlight_location_on_image(self):
        image = self.cv2_color_image
        height, width, channels = image.shape
        first_point = int((int(self.x_entry_widget.get())) / LINE_AMOUNT * width), \
                      int(height - (int(self.y_entry_widget.get()) + 1) / LINE_AMOUNT * height)
        second_point = int((int(self.x_entry_widget.get()) + 1) / LINE_AMOUNT * width), \
                       int(height - (int(self.y_entry_widget.get())) / LINE_AMOUNT * height)
        blurred_image = cv2.GaussianBlur(image, (29, 29), 0)
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.rectangle(mask, first_point,
                             second_point,
                             (255, 255, 255),
                             -1)
        image = np.where(mask == np.array([255, 255, 255]),
                         image,
                         blurred_image)
        image = cv2.rectangle(image, first_point, second_point,
                              (255, 0, 0), 1)
        image = PIL.Image.fromarray(image)
        image = PIL.ImageTk.PhotoImage(image)
        return image

root = tk.Tk()
GraphicalUserInterface(root)
root.mainloop()