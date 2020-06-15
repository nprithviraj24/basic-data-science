import sys
import os
import PIL
from PIL import ImageTk, Image

# from tkinter import *
# window=Tk()# window.title("Running Python Script")
# window.geometry('550x200')
# def run():
#     os.system('python test.py')

# btn = Button(window, text="Click Me", bg="black", fg="white",command=run)
# # btn.grid(column=0, row=0)
# btn.place(x = 50,y = 50)

# text = Text(window)
# text.insert(INSERT, "\n \t This is the review for Macbook pro ")
# text.pack()
# window.mainloop()


from tkinter import *
 
window = Tk()
 
window.title("Recommended Systems")
 
window.geometry('500x600')
 
lbl = Label(window, text="Review for Apple Macbook Pro")
 
lbl.grid(column=5, row=2)
 
# txt = Entry(window,width=10)
 
# txt.grid(column=3, row=3)
 
def clicked():
    import run2 as r
    toD = r.init()
    if toD > 410 :
        lbl = Label(window, text="Reviews are POSITIVE. Hence Product has overall POSITIVE sentiment.") 
        lbl.grid(column=5, row=5)
    else :
        lbl = Label(window, text="Reviews are NEGATIVE. Hence Product has overall NEGATIVE sentiment.")
 
        lbl.grid(column=5, row=5)
    # os.system('python test.py')

    # lbl.configure(text="Button was clicked !!")
    # img = ImageTk.PhotoImage(PIL.Image.open("SVC.png"))
    # panel = Label(window, image = img)
    # # panel.pack(side = "bottom", fill = "both", expand = "yes")
    # panel.grid(column=4, row=3)
 
btn = Button(window, text="Click Me", command=clicked)
 
btn.grid(column=5, row=3)
# img = ImageTk.PhotoImage(PIL.Image.open("SVC.png"))
# panel = Label(window, image = img)
# # panel.pack(side = "bottom", fill = "both", expand = "yes")
# panel.grid(column=4, row=3)
window.mainloop()