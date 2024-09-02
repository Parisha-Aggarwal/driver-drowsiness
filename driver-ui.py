import cv2
import dlib
import numpy as np
from imutils import face_utils
from pygame import mixer
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the mixer for sound alerts
mixer.init()
sound = mixer.Sound('alarm.wav')


# Function to calculate the Euclidean distance between two points
def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Threshold for detecting eye closure
thres = 6
dlist = []


# Function to start the drowsiness detection
def start_detection():
    global dlist
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Calculate distances between eyelid landmarks
            le_38, le_42 = shape[37], shape[41]
            le_39, le_41 = shape[38], shape[40]
            re_44, re_48 = shape[43], shape[47]
            re_45, re_47 = shape[44], shape[46]

            # Append the result of the threshold comparison to dlist
            dlist.append(
                (dist(le_38, le_42) + dist(le_39, le_41) + dist(re_44, re_48) + dist(re_45, re_47)) / 4 < thres)
            if len(dlist) > 10: dlist.pop(0)

            # Draw circles on the landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Drowsiness detected
            if sum(dlist) >= 4:
                try:
                    sound.play()
                except:
                    pass
            else:
                try:
                    sound.stop()
                except:
                    pass

        # Display the frame in the Tkinter window
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        lbl_video.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to stop the drowsiness detection
def stop_detection():
    messagebox.showinfo("Exit", "Stopping the detection.")
    root.quit()


# Setting up the Tkinter window
root = Tk()
root.title("Drowsiness Detection")
root.geometry("800x600")

# Adding a Label widget to display the video feed
lbl_video = Label(root)
lbl_video.pack()

# Adding a Start button to initiate the detection
btn_start = Button(root, text="Start Detection", command=start_detection)
btn_start.pack(side=LEFT, padx=20, pady=20)

# Adding a Stop button to terminate the detection
btn_stop = Button(root, text="Stop Detection", command=stop_detection)
btn_stop.pack(side=RIGHT, padx=20, pady=20)

# Start the Tkinter main loop
root.mainloop()
