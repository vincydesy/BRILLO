from tkinter import *
from tkinter import messagebox

from PIL import ImageTk, Image
from VideoDetection import face_recognition
import numpy as np
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="brillo",
  password="brillo",
  database="brillo"
)
mycursor = mydb.cursor()
np.set_printoptions(threshold=np.inf)

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def check(arr):
    array_img = np.array2string(arr)
    array_img=str(array_img).strip('[ ]')
    array_img=np.fromstring(array_img, dtype=float, sep=" ")
    mycursor.execute("SELECT REPLACE(Chiave, CHAR(10), '') FROM Persona1")
    myresult = mycursor.fetchall()
    i=1
    for faces in myresult:
        faces=str(faces)
        faces = faces.replace(',','')
        faces = faces.replace('[','')
        faces = faces.replace(']','')
        faces = faces.replace('(', '')
        faces = faces.replace(')', '')
        faces=faces.replace("'",'')
        faces = np.fromstring(faces, dtype=float, sep=" ")
        distance = findEuclideanDistance(array_img,faces)
        print(distance)
        if (distance<=0.10):
            print("Identità in DB n° "+str(i))
            return 1;
        i=i+1
    return 0;

def sign():
    name = en.get()
    surname = ec.get()
    cf = ecf.get()
    if (name != '' and surname != '' and cf != ''):
        control,arr=face_recognition()
        if (control==0):
            messagebox.showinfo(title="Errore", message="Errore nel rilevamento del volto, riprovare per favore!")
        else:
            out = check(arr)
            if (out == 0):
                array_img = np.array2string(arr)
                sql = "INSERT INTO Persona1 (Nome, Cognome, CF, Chiave) VALUES (%s, %s, %s, %s)"
                val = (name, surname, cf, array_img)
                mycursor.execute(sql, val)
                mydb.commit()
                messagebox.showinfo(title="Registrazione", message="Registrazione Completata")
            else:
                messagebox.showinfo(title="Attenzione", message="Sei già registrato")
    else:
        messagebox.showinfo(title="Errore", message="Per favore, compila tutti i campi!")

    en.delete(0, END)
    ec.delete(0, END)
    ecf.delete(0, END)


Token = Tk()
F = Frame(Token)
w = Label(Token, text="Benvenuto, registrati qui!")
w.pack()
img = ImageTk.PhotoImage(Image.open("prisca.jpg"))
panel = Label(Token, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")
wn = Label(Token, text="Nome")
wn.pack()
en = Entry(Token, bd =5)
en.pack()
wc = Label(Token, text="Cognome")
wc.pack()
ec = Entry(Token, bd =5)
ec.pack()
wcf = Label(Token, text="Codice Fiscale")
wcf.pack()
ecf = Entry(Token, bd =5)
ecf.pack()
b = Button(Token, text="Registrati!", command = sign)
b.pack()
Token.mainloop()