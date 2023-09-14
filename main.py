import tkinter as tk
import pyautogui
import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk
from numpy import argmax
from keras.src.utils import load_img, img_to_array

root = tk.Tk()
root.title("Aplication for digit recognizing")
root.minsize(600, 500)
root.geometry("600x500+" + str(int(root.winfo_screenwidth()/2 - 300)) + "+" + str(int(root.winfo_screenheight()/2 - 250)))
root.grid_columnconfigure(0, weight = 1)
root.grid_columnconfigure(1, weight = 1)
root.grid_rowconfigure(0, weight = 1)

# LEFT SIDE
leftSide = Frame(root, pady=20)
canvas = Canvas(leftSide, bg='black', height=250, width=250, highlightthickness=1, highlightbackground="black")
canvas.pack()
def getXY(event):
    global canX, canY
    canX, canY = event.x, event.y
    # print("X: " + str(canX) + "    Y: " + str(canY))
def draw(event):
    global canX, canY
    canvas.create_line((canX, canY, event.x, event.y), fill='white', width=8)
    canX, canY = event.x, event.y
def delete(event):
    canvas.delete("all")
    # x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
    # w, h = canvas.winfo_width(), canvas.winfo_height()
    # #pyautogui.screenshot(region=(x, y, w, h)).save('tmp.jpg')
    resetImage()
    lblPrediction.configure(text="Predicted digit: -")
    strChances.set("0: 0%\n1: 0%\n2: 0%\n3: 0%\n4: 0%\n5: 0%\n6: 0%\n7: 0%\n8: 0%\n9: 0%\n")
def save(event):
    x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
    w, h = canvas.winfo_width(), canvas.winfo_height()
    pyautogui.screenshot(region=(x, y, w, h)).save('img/tmp.jpg')
    refresh()
def refresh():
    img = Image.open("img/tmp.jpg")
    imgSmall = img.resize((28, 28), resample=Image.Resampling.BILINEAR)

    result = imgSmall.resize(img.size, Image.Resampling.NEAREST)
    # result = img.resize((28, 28))
    result.save('img/pixelated.jpg')

    imgObj = Image.open("img/pixelated.jpg")
    imgObj = imgObj.resize([250, 250])
    tmpImg = ImageTk.PhotoImage(imgObj)
    label.configure(image=tmpImg)
    label.image = tmpImg

    predict()
def resetImage():
    imgObj = Image.open("img/default.jpg")
    imgObj = imgObj.resize([250, 250])
    tmpImg = ImageTk.PhotoImage(imgObj)
    label.configure(image=tmpImg)
    label.image = tmpImg

global model
model = tf.keras.models.load_model('models/model7.h5')
def predict():
    img = load_img('img/pixelated.jpg', color_mode="grayscale", target_size=(28,28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    prediction = model.predict(img)
    digit = argmax(prediction)
    print(digit)
    chances = prediction[0]
    lblPrediction.configure(text="Predicted digit: " + str(digit))
    refreshPrediction(chances)
def refreshPrediction(chances):
    string = ""
    for i in range(len(chances)):
        string += str(argmax(chances)) + ":  " + str(int(max(chances) * 100)) + "%\n"
        chances[argmax(chances)] = -1

    print(string)
    strChances.set(string)

canvas.bind("<Button-1>", getXY)
canvas.bind("<Button-3>", delete)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", save)

# RIGHT SIDE
rightSide = Frame(root, pady=20)

imgObj = Image.open("img/default.jpg")
imgObj = imgObj.resize([250, 250])
tmpImg = ImageTk.PhotoImage(imgObj)
label = Label(rightSide, image=tmpImg)
label.pack()

leftSide.grid(row = 0, column = 0, sticky = "nesw")
rightSide.grid(row = 0, column = 1, sticky = "nesw")

global strChances
strChances = StringVar()
strChances.set("0: 0%\n1: 0%\n2: 0%\n3: 0%\n4: 0%\n5: 0%\n6: 0%\n7: 0%\n8: 0%\n9: 0%\n")

global lblChances
lblChances = Label(root)
lblChances.grid(row = 1, column=0, sticky = "nesw")
lblChances.configure(textvariable=strChances)

global lblPrediction
lblPrediction = Label(root, text="Predicted digit: -")
lblPrediction.grid(row = 1, column=1, sticky = "nesw")

root.mainloop()