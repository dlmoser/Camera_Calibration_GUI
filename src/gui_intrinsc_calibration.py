from tkinter import *
import PIL
from PIL import Image, ImageTk
import intrinsic_camera_calibration as icc
import tkinter.filedialog as fdialog
import glob
import time
import copy
import os
check_dict = {}


# normal_newmatrix_settings = ['balance',]


class Window(Frame):
    Frame_master = None
    image_path = ""
    params_path = ""
    calibration_images = []
    calibration_flags = {"var_list": None, "calibration_frame": None}
    new_matrix_setting_dic = {"var_dict": None, "new_cammatrix_frame": None}
    currently_working = False

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.rootPath = "/".join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
        self.image_path = self.rootPath+'/CalibrationImages'
        self.params_path = self.rootPath+'/CalibrationParameter'
        # icc.load_images("asd")
        # self.create_clibration_window()
        #self.create_selection_window()
        self.create_path_window()
        #        self.create_clibration_window()

    def create_selection_window(self):
        self.master.title("Mode")
        # self.master.geometry("600x70")
        self.pack(fill=BOTH, expand=YES)

        Frame_master = Frame(self)
        Frame_master.pack(fill=BOTH, expand=YES)

        Settings_Frame1 = Frame(Frame_master)
        Settings_Frame1.pack(fill=BOTH, expand=YES)

        Label(Settings_Frame1, text="Choose if you want record data from a webcam or load calibration images form folder").pack(
            fill=BOTH, anchor=W, expand=YES, ipady=10, ipadx=10)

        Settings_Frame2 = Frame(Frame_master)
        Settings_Frame2.pack(fill=BOTH, expand=YES)

        Button1 = Button(Settings_Frame2, text="Load form Folder",
                         command=self.create_path_window)
        Button1.pack(side=LEFT, anchor=W, expand=YES, fill=BOTH, ipady=5)

        Button2 = Button(Settings_Frame2, text="Capture Images",
                         command=self.create_capture_window)
        Button2.pack(side=LEFT, anchor=W, expand=YES, fill=BOTH, ipady=5)

        self.Frame_master = Frame_master

    def create_capture_window(self):
        if self.Frame_master is not None:
            self.Frame_master.destroy()

        self.master.title("Capture")
        Frame_master = Frame(self)
        Frame_master.pack(fill=BOTH, expand=YES)
        self.pack(fill=BOTH, expand=YES)

        Settings_Frame = Frame(Frame_master)
        Settings_Frame.pack(side=TOP, fill=(X))
        Image_Frame = Frame(Frame_master)
        Image_Frame.pack()
        self.img_label = Label(Image_Frame)
        self.img_label.pack()

        src = makeentry(Settings_Frame, "Video source:", width=3, default=0)

        icc.show_images_from_webcam(self, src=0)

    def create_path_window(self):

        if self.Frame_master is not None:
            self.Frame_master.destroy()

        self.master.title("Mode")
        Frame_master = Frame(self)
        Frame_master.pack(fill=BOTH, expand=YES)

        self.pack(fill=BOTH, expand=YES)

        Settings_Frame1 = Frame(Frame_master)
        Settings_Frame1.pack(fill=BOTH, expand=YES, ipady=7)
        Settings_Frame1.config(relief=GROOVE, bd=2)

        Label(Settings_Frame1, text="Calibration Image Path:").pack(
            side=LEFT, fill=X, anchor=W, padx=20)


        self.Button1_text = StringVar()
        self.Button1_text.set(path_last_dir(self.image_path))
        Button1 = Button(Settings_Frame1, textvariable=self.Button1_text, command=self.set_image_path)
        Button1.pack(side=RIGHT, fill=Y, anchor=W)

        Settings_Frame2 = Frame(Frame_master)
        Settings_Frame2.pack(fill=BOTH, expand=YES, ipady=7)
        Settings_Frame2.config(relief=GROOVE, bd=2)

        Label(Settings_Frame2, text="Store Calibration Parameter:").pack(
            side=LEFT, fill=X, anchor=W, padx=20)

        self.Button2_text = StringVar()
        self.Button2_text.set(path_last_dir(self.params_path))
        Button2 = Button(Settings_Frame2, textvariable=self.Button2_text, command=self.set_parameter_path)
        Button2.pack(side=RIGHT, fill=Y, anchor=W)

        Settings_Frame3 = Frame(Frame_master)
        Settings_Frame3.pack(fill=BOTH, expand=YES, ipady=7)
        Settings_Frame3.config(relief=GROOVE, bd=2)

        Button3 = Button(Settings_Frame3, text="OK", command=self.check_if_path_exist)
        Button3.pack(side=RIGHT, fill=BOTH, expand=YES)
        self.Frame_master = Frame_master
        # filedialog.askdirectory(initialdir='.')

    def create_clibration_window(self):
        global img_label
        global Image_Frame

        if self.Frame_master is not None:
            self.Frame_master.destroy()

        self.Frame_master = Frame(self)
        self.Frame_master.pack(fill=BOTH, expand=YES)
        self.master.title("Calibration")
        self.pack(fill=BOTH, expand=1)

        Settings_Frame = Frame(self.Frame_master)
        Settings_Frame.pack(side=TOP, fill=(X))

        self.Image_Frame = Frame(self.Frame_master)
        self.Image_Frame.pack()

        render = numpy_to_tkimg(icc.calibration_images[0])
        self.img_label = Label(self.Image_Frame, image=render)
        self.img_label.image = render
        self.img_label.pack()

        Settings_Frame_1 = Frame(Settings_Frame)
        L1 = Label(Settings_Frame_1, text="Choose Calibration Model")
        Settings_Frame_2 = Frame(Settings_Frame)
        L2 = Label(Settings_Frame_2, text="Choose calibrate Settings")
        L2.pack(side=TOP, anchor=NW,  expand=NO,  ipadx=7, ipady=7)
        self.calibration_flags["calibration_frame"] = Frame(Settings_Frame_2)
        Settings_Frame_3 = Frame(Settings_Frame)
        L3 = Label(Settings_Frame_3, text="Choose new Cameramatrix settings")
        L3.pack(side=TOP, anchor=NW,  expand=NO,  ipadx=7, ipady=7)
        self.new_matrix_Frame = Frame(Settings_Frame_3)
        Settings_Frame_4 = Frame(Settings_Frame)

        Settings_Frame_1.pack(side=LEFT, fill=BOTH,  expand=NO)
        Settings_Frame_2.pack(side=LEFT, fill=BOTH,  expand=YES)
        Settings_Frame_3.pack(side=LEFT, fill=BOTH,  expand=YES)
        Settings_Frame_4.pack(side=LEFT, fill=BOTH,  expand=NO)
        Settings_Frame_1.config(relief=GROOVE, bd=2)
        Settings_Frame_2.config(relief=GROOVE, bd=2)
        Settings_Frame_3.config(relief=GROOVE, bd=2)
        Settings_Frame_4.config(relief=GROOVE, bd=2)
        self.Settings_Frame_2 = Settings_Frame_2
        self.Settings_Frame_3 = Settings_Frame_3

        L1.pack(side=TOP, anchor=NW, expand=NO, ipadx=7, ipady=7)

        tkvar = StringVar(self)
        choices = {'Fisheye', 'Normal', 'Omnidir'}
        tkvar.set('Normal')  # set the default option

        popupMenu = OptionMenu(Settings_Frame_1, tkvar,  *choices, command=self.dropdown_choise)
        self.dropdown_choise('Normal')
        # chreate_checkbutton('Fisheye')
        popupMenu.pack(side=TOP, anchor=NW, expand=1, fill=BOTH, ipadx=7, ipady=7)

        Button(Settings_Frame_4, text="Calibrate", command=self.start_calibration).pack(
            side=LEFT, anchor=W, fill=BOTH,  ipadx=15, ipady=15)

        # Button2 = Button(Settings_Frame_2, text="bla")
        # Button2.pack(side=LEFT, anchor=W, expand=YES)


    def check_if_path_exist(self):
        if not self.image_path:
            print("Select Image Path")
        elif not self.params_path:
            print("Select Parameter Path")
        else:
            if icc.load_images(self.image_path) == 0:
                print("could not find any images in: {}".format(self.image_path))
                self.create_path_window()
            else:
                self.create_clibration_window()

    def dropdown_choise(self, model):
        self.model = model
        if model == 'Normal':
            calibration_settings = icc.normal_calibrater.calibration_flags
            new_cameramatrix_settings = icc.normal_calibrater.new_camera_matrix_args
        elif model == 'Fisheye':
            calibration_settings = icc.fisheye_calibrater.calibration_flags
            new_cameramatrix_settings = icc.fisheye_calibrater.new_camera_matrix_args
        elif model == 'Omnidir':
            calibration_settings = icc.omnidir_calibrater.calibration_flags
            new_cameramatrix_settings = icc.omnidir_calibrater.new_camera_matrix_args

        self.new_matrix_setting_dic = chreate_new_camera_matrix_settings(
            self.new_matrix_setting_dic['new_cammatrix_frame'], self.Settings_Frame_3, new_cameramatrix_settings, model=model)
        self.calibration_flags = chreate_checkbuttons(
            self.calibration_flags["calibration_frame"], self.Settings_Frame_2, calibration_settings)

    def start_calibration(self):
        if self.currently_working == True:
            print("currently_working")
            return
        self.currently_working = True
        icc.load_images(self.image_path)
        if self.model == 'Normal':
            calib = icc.normal_calibrater(icc.calibration_images)
            calib.set_calibration_flags(get_flaglist_values(self.calibration_flags["var_list"]))
            calib.set_new_cameramatrix_settings(
                get_new_cammatrix_values(self.new_matrix_setting_dic['var_dict']))
            calib.find_chessboard_corners(self)
            calib.calc_calibration_matrix()
            calib.calibrate()
            calib.print_undistoreted_images(self, sleep_time=0.1)
        if self.model == 'Fisheye':
            calib = icc.fisheye_calibrater(icc.calibration_images)
            calib.set_calibration_flags(get_flaglist_values(self.calibration_flags["var_list"]))
            calib.set_new_cameramatrix_settings(
                get_new_cammatrix_values(self.new_matrix_setting_dic['var_dict']))
            calib.find_chessboard_corners(self)
            calib.calc_calibration_matrix()
            calib.calibrate()
            calib.print_undistoreted_images(self, sleep_time=0.1)
            # create_select_string(, fisheye_settings)
            # print(self.calibration_flags["var_list"][0].get())
        if self.model == 'Omnidir':
            calib = icc.omnidir_calibrater(icc.calibration_images)
            calib.set_calibration_flags(get_flaglist_values(self.calibration_flags["var_list"]))
            calib.set_new_cameramatrix_settings(
                get_new_cammatrix_values(self.new_matrix_setting_dic['var_dict']))
            calib.find_chessboard_corners(self)
            calib.calc_calibration_matrix()
            calib.calibrate()
            calib.print_undistoreted_images(self, sleep_time=0.1)
        self.currently_working = False
        del calib
        print("end calibration")

    def set_image_path(self):
        self.image_path = fdialog.askdirectory(initialdir=self.image_path)
        self.Button1_text.set(path_last_dir(self.image_path))

    def set_parameter_path(self):
        self.params_path = fdialog.askdirectory(initialdir=self.params_path)
        self.Button2_text.set(path_last_dir(self.params_path))

    def display_image(self, image):
        print("gui display")
        render = numpy_to_tkimg(image)
        self.img_label.configure(image=render)
        self.img_label.image = render
        self.img_label.update()


def chreate_new_camera_matrix_settings(in_frame, out_frame, new_cam_matrix_settings, model):

    new_cam_matrix_selection = {}
    if in_frame:
        in_frame.destroy()
    in_frame = Frame(out_frame)
    in_frame.pack(expand=YES, fill=BOTH)
    pack_frame = Frame(in_frame)

    pack_frame = Frame(in_frame)
    pack_frame.pack(side=LEFT, fill=Y)

    new_cam_matrix_selection['R'] = create_rotation_matrix_frame(
        pack_frame, default_roll=new_cam_matrix_settings['R'][0], default_pitch=new_cam_matrix_settings['R'][1], default_yaw=new_cam_matrix_settings['R'][2])

    pack_frame = Frame(in_frame)
    pack_frame.pack(side=LEFT, fill=Y)
    new_cam_matrix_selection['new_size'] = create_new_size_frame(
        pack_frame, default=new_cam_matrix_settings['new_size'])

    new_cam_matrix_selection['new_shift'] = create_shift(pack_frame, default=new_cam_matrix_settings['new_shift'])

    if model == "Fisheye":
        pack_frame = Frame(in_frame)
        pack_frame.pack(side=LEFT, fill=Y)
        new_cam_matrix_selection['balance'] = makeentry(
            pack_frame, "balance", 3, default=new_cam_matrix_settings['balance'])
        new_cam_matrix_selection['fov_scale'] = makeentry(
            pack_frame, "fov_scale", 3, default=new_cam_matrix_settings['fov_scale'])
    if model == "Normal":
        pack_frame = Frame(in_frame)
        pack_frame.pack(side=LEFT, fill=Y)
        new_cam_matrix_selection['alpha'] = makeentry(
            pack_frame, "alpha", 3, default=new_cam_matrix_settings['alpha'])
    if model == "Omnidir":
        new_cam_matrix_selection['fov_scale'] = makeentry(
            pack_frame, "fov_scale", 3, default=new_cam_matrix_settings['fov_scale'])
        pack_frame = Frame(in_frame)
        pack_frame.pack(side=LEFT, fill=Y)
        new_cam_matrix_selection['rectify_flag'] = ratdio_bution(
            pack_frame, new_cam_matrix_settings['rectify_flags'])
    return {"var_dict": new_cam_matrix_selection, "new_cammatrix_frame": in_frame}


def ratdio_bution(in_frame, selection_list):
    rectify_flag = StringVar()
    rectify_flag.set(selection_list[0])
    for s in selection_list:
        Radiobutton(in_frame, variable=rectify_flag, value=s,
                    text=s).pack(side=TOP, anchor=W, expand=NO)
    return rectify_flag
#    btOrange = Radiobutton(in_frame, variable=self.colorVar, value="Orange")
#    btPurple = Radiobutton(in_frame, variable=self.colorVar, value="Purple")


def create_new_size_frame(in_frame, default=None):
    #    pack_frame_new_size = Frame(in_frame)
    #    pack_frame_new_size.pack(side=TOP)
    new_size = {}
    new_size['row'] = makeentry(in_frame, "new_size: row", 5, default=default[0])
    new_size['column'] = makeentry(in_frame, "new_size: column", 5, default=default[1])
    return new_size

def create_shift(in_frame, default=None):
    new_shift = {}
    new_shift['row'] = makeentry(in_frame, "shift: row", 5, default=default[0])
    new_shift['column'] = makeentry(in_frame, "shift: column", 5, default=default[1])
    return new_shift

def create_rotation_matrix_frame(in_frame, default_roll=0, default_pitch=0, default_yaw=0):
    #    pack_frame_rotation_matrix = Frame(in_frame)
    #    pack_frame_rotation_matrix.pack(side=TOP)
    R = {}
    R['roll'] = makeentry(in_frame, "R: roll:", 3, default=default_roll)
    R['pitch'] = makeentry(in_frame, "R: pitch:", 3, default=default_pitch)
    R['yaw'] = makeentry(in_frame, "R: yaw:", 3, default=default_yaw)
    return R


def makeentry(parent, caption, width=None, default=None):
    pack_entry = Frame(parent)
    pack_entry.pack(side=TOP, anchor=W, fill=X)
    Label(pack_entry, text=caption).pack(side=LEFT)
    entry = Entry(pack_entry)
    if width:
        entry.config(width=width)
    entry.pack(side=RIGHT)
    if default != None:
        entry.insert(0, default)
    return entry


def chreate_checkbuttons(in_frame, out_frame, flag_list, num_row=4):
    if in_frame:
        in_frame.destroy()
    in_frame = Frame(out_frame)
    in_frame.pack(expand=YES, fill=BOTH)
    var_list = []
    check_list = []
    # check_list.append(in_frame)

    for idx, pick in enumerate(flag_list):
        if (idx) % num_row == 0:
            setting_f = Frame(in_frame)
            setting_f.pack(side=LEFT, fill=Y)
            check_list.append(setting_f)
        var = IntVar()
        chk = Checkbutton(setting_f, text=pick, variable=var)
        chk.pack(side=TOP, anchor=W, expand=NO)
        var_list.append(var)
        # check_list.append(chk)
    return {"var_list": var_list, "calibration_frame": in_frame}


def get_flaglist_values(flag_list):
    select_flag_list = []
    for fl in flag_list:
        select_flag_list.append(bool(fl.get()))
    return select_flag_list


def get_new_cammatrix_values(var_dict):
    #    print("aaaaaaaaaa", var_dict)
    #    var_dict_copy = copy.deepcopy(var_dict)

    var_dict_copy = var_dict.copy()
    for var in var_dict:
        if isinstance(var_dict[var], dict):
            var_dict_copy[var] = var_dict[var].copy()
            for v in var_dict[var]:
                var_dict_copy[var][v] = var_dict_copy[var][v].get()
        else:
            var_dict_copy[var] = var_dict_copy[var].get()
    return var_dict_copy


def Path_select_click():
    file = fdialog.askdirectory()
    # Split the filepath to get the directory

    print(file)


def numpy_to_tkimg(img_npy):
    img = PIL.Image.fromarray(img_npy)
    return ImageTk.PhotoImage(image=img)


def client_exit(asd):
    exit()


def destroy_elements(elements):
    for el in elements:
        el.destroy()

def path_last_dir(string):
    return '/'+string.split('/')[-1]

def truncate_string(string, max_size = 35):
    if len(string) > max_size:
        return "... "+string[-max_size:]
    return string

# root = Tk()
# chreate_clibration_window(root)
# root.mainloop()

root = Tk()
app = Window(root)
root.mainloop()
