# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd
#import datetime
from tkinter import font as tkFont
from tkinter import PhotoImage 
import time
import tkinter.ttk as ttk
import subprocess
import logging
# class Close():        
#     def close_X_Main(self):
#         self.X_Main.destroy()

class X_Main():
    def __init__(self):
        # Create info ui
        #self.master = master
        self.xmain = tk.Tk()
        self.xmain.title("X_Main - @minhcuong-AILab")
        self.xmain.geometry("1300x800")
        # self.xmain.resizable(False, False)
        # Background main x
        self.background_image = PhotoImage(file="uii/main0.png")
        self.background_label = tk.Label(self.xmain, image=self.background_image)
        self.background_label.place(relwidth=1, relheight=1)
        # Main gui
        #self.xmain.attributes('-fullscreen', True) setting fullscreen image
        self.button_X_Register = tk.Button(self.xmain, text="ĐĂNG KÍ",command=self.open_X_Register,fg="white",anchor='w',justify='left',borderwidth=0  ,width=9  ,height=1 , activebackground = "Red" ,font=('Helvetica', 15, ' bold '), background="#07080A")#.grid(row=8, column=2, padx=5)
        self.button_X_Register.place(x=310, y=193)
        self.button_X_Register = tk.Button(self.xmain, text="HUẤN LUYỆN",command=self.open_X_Training,fg="white",anchor='w',justify='left',borderwidth=0  ,width=11  ,height=1 , activebackground = "Red" ,font=('Helvetica', 15, ' bold '), background="#07080A")#.grid(row=8, column=2, padx=5)
        self.button_X_Register.place(x=310, y=310)
        self.button_X_Recognition = tk.Button(self.xmain, text="NHẬN DIỆN",command=self.open_X_Recog,anchor='w',justify='left' ,fg="white",borderwidth=0    ,width=9  ,height=1, activebackground = "Red" ,font=('Helvetica', 15, ' bold '), background="#07080A")
        self.button_X_Recognition.place(x=310, y=430)
        self.button_X_Out = tk.Button(self.xmain, text="THOÁT",command=self.out_X_Main,anchor='w',justify='left' ,fg="white",borderwidth=0    ,width=9  ,height=1, activebackground = "Red" ,font=('Helvetica', 15, ' bold '), background="#07080A")
        self.button_X_Out.place(x=310, y=548)
    # Open X_Register Part---
    def open_X_Register(self):
        print("******      X_Register @minhcuong-AILab Already!")
        self.X_Register_Path = "X_Register.py"
        subprocess.call(['python3', self.X_Register_Path])
    # Open X_Training Part---
    def open_X_Training(self):
        print("******      X_Training @minhcuong-AILab Already!")
        self.X_Training_Path = "X_Training.py"
        subprocess.call(['python3', self.X_Training_Path])
    def open_X_Recog(self):
        print("******      X_Recog_OT @minhcuong-AILab Already!")
        self.X_Recog_OT_Path = "X_Recog_OT.py"
        subprocess.call(['python3', self.X_Recog_OT_Path])
    # Out X_Main Program---
    def out_X_Main(self):
        self.xmain.destroy()
        print("******      X_Main - @minhcuong-AILab ~.~ Hope you enjoy that!!!")
    def run(self):
        self.xmain.mainloop()


def main():
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.debug("Nhớ debug nha!!!")
    logging.info("Check kĩ thông tin đi")
    logging.warning("Đây là cảnh báo cho bạn nè!!!")
    logging.error("Lỗi chương trình rồi bạn ơi!!!")
    logging.critical("Bạn đã làm cái gì vậy?")
    running = X_Main()
    running.run()

if __name__ == "__main__":
    main()





