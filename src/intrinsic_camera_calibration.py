import sys
sys.path.append("/home/david/Documents/BSB/Software/utils/")
#from web_socket import websocket_func, stop_websocket
#from json_utils import GetImageFromJson
import cv2
import time
import numpy as np
import glob
import threading
from transformations import *
cam = "left"
image_buffer = []
from CamVideoStream import CamStream


class intrinsic_calibration:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    selected_calibraiton_flags = 0
    R = np.eye(3, 3)
    D = np.zeros((4, 1))
    K = np.eye(3, 3)  # Camera matrix
    K[0, 2] = 1920 / 2
    K[1, 2] = 1080 / 2
    xi = None

    def __init__(self, images):
        self.images = images
        self.img_shape = images[0].shape[:2]
        self.img_shape_flip = (self.img_shape[1], self.img_shape[0])
        self.objp = np.zeros((1, 6 * 8, 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 25
        self.objp[0, :, 0:1] = self.objp[0, :, 0:1] - self.img_shape[0] / 2
        self.objp[0, :, 1:2] = self.objp[0, :, 1:2] - self.img_shape[1] / 2
        self.objpoints = []
        self.imgpoints = []
        self.status = 0

    def init_parameter(self):
        self.R = np.eye(3, 3)
        self.D = np.zeros((4, 1))
        self.K = np.eye(3, 3)  # Camera matrix
        self.K[0, 2] = 1920 / 2
        self.K[1, 2] = 1080 / 2
        self.K[0,0]=500
        self.K[1,1]=500

    def find_chessboard_corners(self, gui=None):  # display = cv2/gui
        if self.status < 0:
            return
        try:
            for img in self.images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
                if ret == True:
                    self.objpoints.append(self.objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    self.imgpoints.append(corners2.reshape(1, -1, 2))
                    img = cv2.drawChessboardCorners(img.copy(), (8, 6), corners2, ret)
                    if gui == 'None':
                        cv2.imshow('img', img)
                        cv2.waitKey(100)
                    else:
                        gui.display_image(img)
                    self.img_len = len(self.objpoints)
            cv2.destroyAllWindows()
        except:
            self.status = -3
            print("Error in find_chessboard_corners")

    def calibrate(self):
        if self.status < 0:
            return
        try:
            self.img_undistorted = []
            for img_distorted in self.images:
                self.img_undistorted.append(cv2.remap(img_distorted, self.map1, self.map2,
                                                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        except Exception as e:
            self.status = -6
            print("Error in calibrate: {}".format(e))

    def print_undistoreted_images(self, gui=None, sleep_time=0.5):
        if self.status < 0:
            return
        for img in self.img_undistorted:
            if gui == 'None':
                cv2.imshow('img', img)
                cv2.waitKey(sleep_time)
            else:
                gui.display_image(img)
                time.sleep(sleep_time)

    def print_one_undistorted_image(self, img):
        if self.status < 0:
            return
        cv2.imshow('img', img)
        cv2.waitKey(0)

    def set_R(self, a_x=0, a_y=0, a_z=0):
        self.R = R_z(g2r(a_z)).dot(R_y(g2r(a_y)).dot(R_x(g2r(a_x))))

    def set_Knew(self, Knew):
        self.Knew = Knew

    def save_params(self):
        cal_dic = {"Knew": self.Knew, "D": self.D, "K": self.K, "xi": self.xi}
        np.save("clibration_params.npy", cal_dic)

    def __exit__(self, exc_type, exc_value, traceback):
        for file in self.files:
            os.unlink(file)


class omnidir_calibrater(intrinsic_calibration):
    calibration_flags = ['FIX_SKEW', 'FIX_K1', 'FIX_K2', 'FIX_P1',
                         'FIX_P2', 'FIX_XI', 'FIX_GAMMA', 'FIX_CENTER']
    new_camera_matrix_args = {
        'R': [0, 0, 0], 'new_size': ['i_r', 'i_c'], 'fov_scale': 2, 'rectify_flags': ['RECTIFY_PERSPECTIVE', 'RECTIFY_CYLINDRICAL', 'RECTIFY_STEREOGRAPHIC', 'RECTIFY_LONGLATI'], "new_shift": [0, 0]}

    # flag = (cv2.omnidir.CALIB_FIX_CENTER + cv2.omnidir.RECTIFY_PERSPECTIVE)

    def __init__(self, images):
        super().__init__(images)
        self.new_camera_matrix_args = fisheye_calibrater.new_camera_matrix_args.copy()


    def set_calibration_flags(self, select_flag_list):
        if self.status < 0:
            return
        try:
            for cf, sfl in zip(self.calibration_flags, select_flag_list):
                if sfl == True:
                    self.selected_calibraiton_flags += eval('cv2.omnidir.CALIB_' + cf)
        except:
            self.status = -1
            print("Error in set_calibration_flags")

    def set_new_cameramatrix_settings(self, select_setting_list):
        if self.status < 0:
            return
        try:
            self.new_camera_matrix_args['R'] = Rrpy(
                roll=float(select_setting_list['R']['roll']), pitch=float(select_setting_list['R']['pitch']), yaw=float(select_setting_list['R']['yaw']), mode='degree')
            if select_setting_list['new_size']['row'] == 'i_r':
                self.new_camera_matrix_args['new_size'][1] = self.img_shape_flip[1]
            else:
                self.new_camera_matrix_args['new_size'][1] = int(
                    select_setting_list['new_size']['row'])
            if select_setting_list['new_size']['column'] == 'i_c':
                self.new_camera_matrix_args['new_size'][0] = self.img_shape_flip[0]
            else:
                self.new_camera_matrix_args['new_size'][0] = int(
                    select_setting_list['new_size']['column'])

            self.new_camera_matrix_args['new_size'] = tuple(self.new_camera_matrix_args['new_size'])
            self.new_camera_matrix_args['fov_scale'] = float(select_setting_list['fov_scale'])
            self.new_camera_matrix_args['rectify_flag'] = select_setting_list['rectify_flag']
        except Exception as e:
            self.status = -2
            print("Error in set_new_cameramatrix_settings: {}".format(e))

    def calc_calibration_matrix(self):
        if self.status < 0:
            return
        try:
            self.init_parameter()
            retval, self.K, self.xi, self.D, rvecs, tvecs, idx = cv2.omnidir.calibrate(self.objpoints, self.imgpoints,
                                                                                       self.img_shape_flip, None, None, None, flags=self.selected_calibraiton_flags, criteria=self.criteria)

            if self.new_camera_matrix_args['rectify_flag'] == 'RECTIFY_PERSPECTIVE':
                self.Knew = np.array([[self.img_shape_flip[0] / 4 * self.new_camera_matrix_args['fov_scale'], 0, self.img_shape_flip[0] / 2],
                                      [0, self.img_shape_flip[1] / 4 *
                                          self.new_camera_matrix_args['fov_scale'], self.img_shape_flip[1] / 2],
                                      [0, 0, 1]])
            else:
                self.Knew = np.array([[self.img_shape_flip[0] / np.pi * self.new_camera_matrix_args['fov_scale'], 0, 0],
                                      [0, self.img_shape_flip[1] / np.pi *
                                          self.new_camera_matrix_args['fov_scale'], 0],
                                      [0, 0, 1]])

            self.map1, self.map2 = cv2.omnidir.initUndistortRectifyMap(
                self.K, self.D, self.xi, self.new_camera_matrix_args['R'], self.Knew, self.new_camera_matrix_args['new_size'], cv2.CV_16SC2, flags=eval('cv2.omnidir.' + self.new_camera_matrix_args['rectify_flag']))
        except Exception as e:
            self.status = -4
            print("Error in calc_calibration_matrix: {}".format(e))


class normal_calibrater(intrinsic_calibration):
    calibration_flags = ['USE_INTRINSIC_GUESS', 'FIX_PRINCIPAL_POINT', 'FIX_ASPECT_RATIO',  'ZERO_TANGENT_DIST', 'FIX_K1', 'FIX_K2', 'FIX_K3',
                         'FIX_K4', 'FIX_K5', 'FIX_K6', 'RATIONAL_MODEL', 'THIN_PRISM_MODEL', 'FIX_S1_S2_S3_S4', 'TILTED_MODEL', 'FIX_TAUX_TAUY']
    new_camera_matrix_args = {
        'R': [0, 0, 0], 'new_size': ['i_r', 'i_c'], 'alpha': 0.5, "new_shift": [0, 0]}

    def __init__(self, images):
        super().__init__(images)
        self.new_camera_matrix_args = fisheye_calibrater.new_camera_matrix_args.copy()

    def set_calibration_flags(self, select_flag_list):
        if self.status < 0:
            return
        try:
            for cf, sfl in zip(self.calibration_flags, select_flag_list):
                if sfl == True:
                    self.selected_calibraiton_flags += eval('cv2.CALIB_' + cf)
        except Exception as e:
            self.status = -1
            print("Error in set_calibration_flags: {}".format(e))

    def set_new_cameramatrix_settings(self, select_setting_list):
        if self.status < 0:
            return
        try:
            self.new_camera_matrix_args['R'] = Rrpy(
                roll=float(select_setting_list['R']['roll']), pitch=float(select_setting_list['R']['pitch']), yaw=float(select_setting_list['R']['yaw']), mode='degree')
            if select_setting_list['new_size']['row'] == 'i_r':
                self.new_camera_matrix_args['new_size'][1] = self.img_shape_flip[1]
            else:
                self.new_camera_matrix_args['new_size'][1] = int(
                    select_setting_list['new_size']['row'])
            if select_setting_list['new_size']['column'] == 'i_c':
                self.new_camera_matrix_args['new_size'][0] = self.img_shape_flip[0]
            else:
                self.new_camera_matrix_args['new_size'][0] = int(
                    select_setting_list['new_size']['column'])

            self.new_camera_matrix_args['new_size'] = tuple(self.new_camera_matrix_args['new_size'])
            self.new_camera_matrix_args['alpha'] = float(select_setting_list['alpha'])
        except Exception as e:
            self.status = -1
            print("Error in set_new_cameramatrix_settings: {}".format(e))

    def calc_calibration_matrix(self):
        try:
            self.init_parameter()
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, self.img_shape_flip, None, None, flags=self.selected_calibraiton_flags)

            self.Knew, self.ROI = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, self.img_shape_flip, self.new_camera_matrix_args['alpha'], self.img_shape_flip)

            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, self.new_camera_matrix_args['R'], self.Knew, self.new_camera_matrix_args['new_size'], cv2.CV_16SC2)
        except Exception as e:
            self.status = -4
            print("Error in calc_calibration_matrix: {}".format(e))


class fisheye_calibrater(intrinsic_calibration):
    calibration_flags = ['USE_INTRINSIC_GUESS', 'RECOMPUTE_EXTRINSIC', 'CHECK_COND', 'FIX_SKEW ',
                         'FIX_K1', 'FIX_K2', 'FIX_K3', 'FIX_K4', 'FIX_INTRINSIC', 'FIX_PRINCIPAL_POINT']

    new_camera_matrix_args = {
        'R': [0, 0, 0], 'new_size': ['i_r', 'i_c'], 'balance': 0.5, 'fov_scale': 1, "new_shift": [0, 0]}

    def __init__(self, images):
        super().__init__(images)
        self.new_camera_matrix_args = fisheye_calibrater.new_camera_matrix_args.copy()
        # self.select_flags =


    def set_calibration_flags(self, select_flag_list):
        if self.status < 0:
            return
        try:
            for cf, sfl in zip(self.calibration_flags, select_flag_list):
                if sfl == True:
                    self.selected_calibraiton_flags += eval('cv2.fisheye.CALIB_' + cf)
        except:
            self.status = -1
            print("Error in set_calibration_flags")

    def set_new_cameramatrix_settings(self, select_setting_list):
        if self.status < 0:
            return
        try:
            self.new_camera_matrix_args['R'] = Rrpy(
                roll=float(select_setting_list['R']['roll']), pitch=float(select_setting_list['R']['pitch']), yaw=float(select_setting_list['R']['yaw']), mode='degree')
            if select_setting_list['new_size']['row'] == 'i_r':
                self.new_camera_matrix_args['new_size'][1] = self.img_shape_flip[1]
            else:
                self.new_camera_matrix_args['new_size'][0] = int(
                    select_setting_list['new_size']['row'])
            if select_setting_list['new_size']['column'] == 'i_c':
                self.new_camera_matrix_args['new_size'][0] = self.img_shape_flip[0]
            else:
                self.new_camera_matrix_args['new_size'][1] = int(
                    select_setting_list['new_size']['column'])

            self.new_camera_matrix_args['new_size'] = tuple(self.new_camera_matrix_args['new_size'])
            self.new_camera_matrix_args['balance'] = float(select_setting_list['balance'])
            self.new_camera_matrix_args['fov_scale'] = float(select_setting_list['fov_scale'])

            self.new_camera_matrix_args['new_shift'][0] = int(select_setting_list['new_shift']['row'])
            self.new_camera_matrix_args['new_shift'][1] = int(select_setting_list['new_shift']['column'])

        except Exception as e:
            print(e)
            self.status = -2
            print("error in set_new_cameramatrix_settings: {}".format(e))

    def calc_calibration_matrix(self):
        if self.status < 0:
            return
        try:
            self.init_parameter()
            self.rvecs = [np.zeros((1, 1, 3), dtype=np.float64)
                          for i in range(self.img_len)]  # Output vector of rotation vectors
            self.tvecs = [np.zeros((1, 1, 3), dtype=np.float64)
                          for i in range(self.img_len)]  # Output vector of translation vectors

            self.rms, self.K, w, e, f = cv2.fisheye.calibrate(
                self.objpoints, self.imgpoints,  self.img_shape_flip, self.K, self.D, self.rvecs, self.tvecs, flags=self.selected_calibraiton_flags)

            self.Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D,  self.img_shape_flip, self.R, balance=self.new_camera_matrix_args['balance'],
                                                                               new_size=self.img_shape_flip, fov_scale=self.new_camera_matrix_args['fov_scale'])

            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, self.new_camera_matrix_args['R'], self.Knew, self.new_camera_matrix_args['new_size'], cv2.CV_16SC2)
            self.map1 = self.map1+self.new_camera_matrix_args['new_shift'][1]
            self.map2 = self.map2+self.new_camera_matrix_args['new_shift'][0]
            print("Knew", self.Knew, "K", self.K, "D", self.D, "map1", self.map1, "map2", self.map2)
            np.save("calibration_params.npy", {"Knew": self.Knew, "K": self.K, "D": self.D, "map1": self.map1, "map2": self.map2})
        except Exception as e:
            self.status = -4
            print("Error in calibraiton Matrix: {}".format(e))

    def print_info(self):
        print('objectp:', self.objp)
        print('imagepoints', self.imgpoints)
        print('rvecs:', self.rvecs)
        print('tvecs:', self.tvecs)
        print("K:", self.K)
        print("D:", self.D)
        print("Knew:", self.Knew)
        print('R:', self.R)
        print('imgsize:', self.img_shape)
        print('Re-projection error reported by calibrateCamera:', self.rms)

    def set_select_flags(self, select_flag_list):
        for cf, sfl in zip(self.calibration_flags, select_flag_list):
            if sfl == True:
                self.selected_calibraiton_flags += eval('cv2.fisheye.CALIB_' + cf)


def find_chessboard_corners_in_image(img, chessboard_layout = (8, 6)):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_layout, None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_corners = cv2.drawChessboardCorners(img.copy(), chessboard_layout, corners2, ret)
        return img_corners, img
    return None, img



def show_images_from_webcam(gui, src=0):
    global calibration_images
    calibration_images = []
    cam = CamStream().create(src)
    img_corners = None
    if cam == None:
        return
    cam.start_stream()
    t1 = time.time()
    while True:
        img = cv2.cvtColor(cam.read(), cv2.COLOR_BGR2RGB)
        t2 = time.time()
        if t2 - t1 > 1:
            img_corners, img = find_chessboard_corners_in_image(img)
            img_corners = img
            if img_corners is not None:
                gui.display_image(image_effect(img_corners))
                time.sleep(2)
            t1 = time.time()
        else:
            gui.display_image(img)


def image_effect(img):
    img = img * 1.5
    img[img > 255] = 255
    return img.astype(np.uint8)


def save_images(calibration_images):
    save_path = '/home/david/Documents/BSB/Software/budding/camera_calibration/calibrate_intrinsics/images/'
    img_num = []
    imgs = sorted(glob.glob(save_path + '*.jpg'))
    if imgs:
        for i in imgs:
            img_num.append(i.split('/')[-1].split('.')[0])
        img_num = np.array(img_num, dtype=int)
        k = np.amax(img_num)
    else:
        k = 1
    for idx, img in enumerate(calibration_images):
        cv2.imwrite(save_path + str(k) + '.jpg', img)
        k += 1


def load_images(path):
    #path = "/home/david/Documents/BSB/Software/camera/camera_calibration/calibrate_intrinsics/images"
    global cut
    global calibration_images
    calibration_images = []
    imgs = sorted(glob.glob(path + '/*jpg'))
    print(imgs)
    for img in imgs:
        calibration_images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
#    print("imgs", calibration_images)
    return len(calibration_images)


def get_images(pipelineJson, wsOutcomeSink):
    imgl, imgr = get_images_form_websocket(pipelineJson, wsOutcomeSink)
    if cam == "left":
        record_images(imgl)
    if cam == "right":
        record_images(imgr)


def intrinsic_calibration(images):
    calib = fisheye_calibrater(images)

    calib.find_chessboard_corners()
    calib.set_R(a_y=45)
    calib.calc_calibration_matrix()
    calib.calibrate()
    calib.print_undistoreted_images()
    calib.print_one_undistorted_image(calib.img_undistorted[0])
#    calib.print_info()
#    calib.save_params()



if __name__ == "__main__":
    global calibration_images
    # model = "fisheye"  # noraml
    # websocket_func(get_images)
    load_images(path)
    intrinsic_calibration(calibration_images)
