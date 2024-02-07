from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import tkinter as tk
from tkinter import filedialog


# Designing window for login

def login():
    # Load the configuration and model weights
    cfg = get_cfg()
    cfg.merge_from_file("C:/Users/imgod/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    predictor = DefaultPredictor(cfg)

    # Open a file dialog to select an image
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    # Read the image
    img = cv2.imread(file_path)

    # Make a prediction on the image
    outputs = predictor(img)

    # Visualize the prediction
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Object Detection", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()






    
def register():
    # Load the configuration and model weights
    cfg = get_cfg()

    cfg.merge_from_file("C:/Users/imgod/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    predictor = DefaultPredictor(cfg)

    # Open the webcam
    
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # Make a prediction on the frame
        outputs = predictor(frame)
        
        # Visualize the prediction
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Object Detection",v.get_image()[:, :, ::-1])
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #while True:
            
            #cv.NamedWindow(window_name, cv.CV_WINDOW_AUTOSIZE)
            #frame = cv.QueryFrame(capture)
       

           # _, frame = cap.read()

            #cv2.imshow(namedwindow, frame)
            #keyCode = cv2.waitKey(1)

            #if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <1:
                #break
        #cv2.destroyAllWindows()

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()




    





# Designing Main(first) window

def main_account_screen():
    global main_screen
    main_screen = Tk()
    main_screen.geometry("500x350")
    main_screen.title("Object Detection")
    Label(text="Select Your Choice", bg="grey", width="300", height="2", font=("Calibri", 13)).pack()
    Label(text="").pack()
    ku=Button(main_screen,text="Object detection using still image", height="2", width="30", command=login)
    ku.pack()
    Label(text="").pack()
    lu=Button(main_screen,text="Object detection using real time image", height="2", width="30", command=register)
    lu.pack()
    def ku_hover(e):
        ku["bg"] = "white"

    def ku_hover_leave(e):
        ku["bg"] = "SystemButtonFace"

    ku.bind("<Enter>", ku_hover)
    ku.bind("<Leave>", ku_hover_leave)
    
    def lu_hover(e):
        lu["bg"] = "white"

    def lu_hover_leave(e):
        lu["bg"] = "SystemButtonFace"

    lu.bind("<Enter>",lu_hover)
    lu.bind("<Leave>", lu_hover_leave)

    main_screen.mainloop()



main_account_screen()


