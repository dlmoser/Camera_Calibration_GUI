import os

# import the necessary packages
import threading
import cv2
import time
import glob


class CamStream:
    def __init__(self):
        self.stop_all = False
        pass

    def create(self, src=0):
        try:
            self.stream = cv2.VideoCapture(src)
            flag, _ = self.stream.read()
            if flag == False:
                raise
            print("CAMSTREAM: {} detected".format(src))
            time.sleep(1)
            return self
        except:
            print("WARNING: CAMSTREAM: {} seems not to work".format(src))
            return None

    def start_stream(self):
        stop_all = False
        t = threading.Thread(target=self.update)
        t.start()
        time.sleep(1)

    def update(self):
        while True:
            if self.stop_all == True:
                break
            _, self.frame = self.stream.read()

    def stop_stream(self):
        self.stop_all = True

    def get_resolution(self):
        print("asd")
        cam1_res = (self.stream.get(cv2.CAP_PROP_FRAME_WIDTH),
                    self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cam1_res

    def set_resolution(self, resolution):
        width, height = resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def grap_frame(self):
        self.retval_g = self.stream.grab()

    def retrieve_frame(self):
        self.retval_r, self.frame = self.stream.retrieve()

    def read_frame(self):
        (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def set_h264(self):
        codec = 1196444237.0  # MJPG
        codec = 844715353.0  # YUY2
        self.stream.set(cv2.CAP_PROP_FOURCC, codec)

    def print_all_infos(self):
        print('cv2.CAP_PROP_POS_MSEC: {}'.format(self.stream.get(cv2.CAP_PROP_POS_MSEC)))
        print('cv2.CAP_PROP_POS_FRAMES: {}'.format(self.stream.get(cv2.CAP_PROP_POS_FRAMES)))
        print('cv2.CAP_PROP_POS_AVI_RATIO: {}'.format(self.stream.get(cv2.CAP_PROP_POS_AVI_RATIO)))
        print('cv2.CAP_PROP_FRAME_WIDTH: {}'.format(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print('cv2.CAP_PROP_FRAME_HEIGHT: {}'.format(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print('cv2.CAP_PROP_FPS: {}'.format(self.stream.get(cv2.CAP_PROP_FPS)))
        print('cv2.CAP_PROP_FOURCC: {}'.format(self.stream.get(cv2.CAP_PROP_FOURCC)))
        print('cv2.CAP_PROP_FRAME_COUNT: {}'.format(self.stream.get(cv2.CAP_PROP_FRAME_COUNT)))
        print('cv2.CAP_PROP_FORMAT: {}'.format(self.stream.get(cv2.CAP_PROP_FORMAT)))
        print('cv2.CAP_PROP_MODE: {}'.format(self.stream.get(cv2.CAP_PROP_MODE)))
        print('cv2.CAP_PROP_BRIGHTNESS: {}'.format(self.stream.get(cv2.CAP_PROP_BRIGHTNESS)))
        print('cv2.CAP_PROP_CONTRAST: {}'.format(self.stream.get(cv2.CAP_PROP_CONTRAST)))
        print('cv2.CAP_PROP_SATURATION: {}'.format(self.stream.get(cv2.CAP_PROP_SATURATION)))
        print('cv2.CAP_PROP_HUE: {}'.format(self.stream.get(cv2.CAP_PROP_HUE)))
        print('cv2.CAP_PROP_GAIN: {}'.format(self.stream.get(cv2.CAP_PROP_GAIN)))
        print('cv2.CAP_PROP_EXPOSURE: {}'.format(self.stream.get(cv2.CAP_PROP_EXPOSURE)))
        print('cv2.CAP_PROP_CONVERT_RGB: {}'.format(self.stream.get(cv2.CAP_PROP_CONVERT_RGB)))
    #    print('cv2.CAP_PROP_WHITE_BALANCE_U: {}'.format(
    #        self.stream.get(cv2.CAP_PROP_WHITE_BALANCE_U)))
    #    print('cv2.CAP_PROP_WHITE_BALANCE_V: {}'.format(
    #        self.stream.get(cv2.CAP_PROP_WHITE_BALANCE_V)))
        print('cv2.CAP_PROP_ISO_SPEED: {}'.format(self.stream.get(cv2.CAP_PROP_ISO_SPEED)))
        print('cv2.CAP_PROP_BUFFERSIZE: {}'.format(self.stream.get(cv2.CAP_PROP_BUFFERSIZE)))


def test():
    cam = CamStream().create()
    print(cam.get_resolution())
    cam.print_all_infos()
