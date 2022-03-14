import tkinter as tk
import tkinter.filedialog
import numpy as np
import numpy.linalg as LA
import random
import datetime
from scipy.io.wavfile import read, write
from PIL import Image, ImageOps
import os

# make new data whose average is zero
def make_new_data(data,ave_data):
  new_data = []
  for i in range(len(data)):
    new_data.append(data[i]-ave_data)
  return new_data

# repetition algorithm
def repetition(w,z):
  w_before = w
  w = w*(1/LA.norm(w))
  max_loop_size = 100
  while (max_loop_size > 0):
    w_before = w
    temp1 = z.copy()
    temp2 = np.dot(w.T,z)
    for i in range(len(temp2)):
      num = temp2[i]
      temp2[i] = pow(num, 3)
    for i in range(len(w)):
      for j in range(len(temp2)):
        temp1[i][j] = z[i][j]*temp2[j]
    w = np.mean(temp1,axis=1) - 3*w
    w = w*(1/LA.norm(w))
    count = 0
    for i in range(len(w)):
      if (abs(abs(w_before[i]) - abs(w[i]))<0.00001):
        count += 1
    if (count == len(w)):
      break
    max_loop_size -= 1
  return w

# read data from file(wav)
def read_file_wav(filenames, process_message_text):
  rates = []
  data = []
  for f in filenames:
    try:
      tmp_rate, tmp_data = read(f)
      rates.append(tmp_rate)
      data.append(tmp_data)
    except:
      process_message_text.set('Process suspended by something error.\nCaution : please select only one type file (ex : only png gray)')
      return
  return data, rates

# read data from file(png gray)
def read_file_png_gray(filenames, process_message_text):
  images = []
  for f in filenames:
    try:
      images.append(np.array(Image.open(f).convert('L')))
    except:
      process_message_text.set('Process suspended by something error.\nCaution : please select only one type file (ex : only png gray)')
      return
  data = []
  for i in range(len(filenames)):
    data.append(images[i].copy())
  for i in range(len(filenames)):
    data[i] = np.ravel(data[i])
  return data, images

# write data(wav)
def write_file_wav(y, RATE, N, write_dir):
  date_now = datetime.datetime.now().isoformat()
  s = []
  for i in range(N):
    s.append(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'.wav')
  for i in range(N):
    write(s[i], RATE, y[i])

# write data(png gray)
def write_file_png_gray(images_new, N, write_dir):
  images = []
  for i in range(N):
    images.append(Image.fromarray(images_new[i].astype(np.uint8)))
  date_now = datetime.datetime.now().isoformat()
  s = []
  for i in range(N):
    s.append(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'.png')
  for i in range(N):
    images[i].save(s[i])
  # convert black & white
  for i in range(N):
    image_inv = ImageOps.invert(images[i])
    image_inv.save(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'_inv.png')


class Application(tk.Frame):
  def __init__(self, root=None):
    super().__init__(root, width=800, height=600)
    self.root = root
    self.pack()

    #file names array
    self.new_file_names_arr = [] # file names added by select btn
    self.file_names_arr = [] # file names to sep

    #file open btn
    fileopen_btn = tk.Button(self, text='Select file', bg="#ffffff", activebackground="#a9a9a9", relief='raised')
    fileopen_btn.bind('<ButtonPress>', self.file_dialog)
    fileopen_btn.place(x=20, y=20, width=150, height=20)

    #run ICA btn (wav)
    self.exe_btn_wav = tk.Button(self, text='Run (wav)', bg="#ffffff", activebackground="#a9a9a9", relief='raised', state=tk.DISABLED)
    #self.exe_btn_wav.bind('<ButtonPress>', self.save_dir_dialog)
    self.exe_btn_wav.bind('<ButtonPress>', self.ica_wav)
    self.exe_btn_wav.place(x=190, y=20, width=150, height=20)

    #run ICA btn (png gray)
    self.exe_btn_png_g = tk.Button(self, text='Run (png gray)', bg="#ffffff", activebackground="#a9a9a9", relief='raised', state=tk.DISABLED)
    #self.exe_btn_png_g.bind('<ButtonPress>', self.save_dir_dialog)
    self.exe_btn_png_g.bind('<ButtonPress>', self.ica_png_gray)
    self.exe_btn_png_g.place(x=360, y=20, width=150, height=20)

    # message
    self.message = 'File unselected'
    self.message_label = tk.Label(text=self.message)
    self.message_label.place(x=40, y=40, width=700, height=20)

    # del_btns
    self.del_btns = []

    # file name labels
    self.file_name_labels = []

    # read data & save data directory
    self.read_dir = './'
    self.write_dir = './'

    # canvas
    """
    self.canvas = tk.Canvas(self, width=700, height=460, bg='#D0A8F4', relief=tk.GROOVE)
    self.canvas.place(x=50, y=60)
    """

    # state while processing mesage
    self.process_message_text = tk.StringVar()
    self.process_message_text.set('After files which you want to separate are selected, please push Run Button.\nCaution : please select only one type file (ex : only png gray)')
    self.process_message = tk.Label(textvariable=self.process_message_text)
    self.process_message.place(x=40, y=540, width=700, height=40)

  # make file name labels
  def make_file_name_labels(self):
    for i in range(len(self.file_names_arr)):
      self.file_name_labels.append(tk.Label(text=self.file_names_arr[i], anchor='w'))
      self.file_name_labels[i].place(x=50, y=60+20*i, width=700, height=20)
  
  # clear all file name labels
  def clear_file_name_labels(self):
    for i in range(len(self.file_names_arr)):
      self.file_name_labels[i].destroy()
    self.file_name_labels = []

  # make file name delete btn
  def make_del_btn(self):
    for i in range(len(self.file_names_arr)):
      self.del_btns.append(tk.Button(self.root, text=str(i+1), bg="#ffffff", activebackground="#a9a9a9", relief='raised'))
      self.del_btns[i].bind('<ButtonPress>', self.del_label)
      self.del_btns[i].place(x=20, y=60+20*i, width=20, height=20)
  
  # clear all file name delete btn
  def clear_del_btn(self):
    for i in range(len(self.file_names_arr)):
      self.del_btns[i].destroy()
    self.del_btns = []
      
  # pushed del_btn
  def del_label(self, event):
    self.process_message_text.set('After files which you want to separate are selected, please push Run Button.\nCaution : please select only one type file (ex : only png gray)')
    index = int(event.widget["text"])-1

    self.clear_del_btn()
    self.clear_file_name_labels()

    self.file_names_arr.pop(index)

    self.change_message_label()
    self.make_del_btn()
    self.make_file_name_labels()

  # update file_names_arr(by fileopen_btn)
  def update_file_names_arr(self):
    for f in self.new_file_names_arr:
      flag = True
      for fo in self.file_names_arr:
        if (f == fo):
          flag = False
          break
      if (flag):
        self.file_names_arr.append(f)
  
  # change message_label & exe_btn state
  def change_message_label(self):
    if (len(self.file_names_arr)==0):
      self.message = 'File unselected'
      self.exe_btn_wav["state"] = tk.DISABLED
      self.exe_btn_png_g["state"] = tk.DISABLED
    else:
      self.message = 'If you want to delete file, please click the number button on the left side of file name'
      if (len(self.file_names_arr)==1):
        self.exe_btn_wav["state"] = tk.DISABLED
        self.exe_btn_png_g["state"] = tk.DISABLED
      else:
        self.exe_btn_wav["state"] = tk.NORMAL
        self.exe_btn_png_g["state"] = tk.NORMAL
    self.message_label['text'] = self.message

  # pushed select file btn
  def file_dialog(self, event):
    self.process_message_text.set('After files which you want to separate are selected, please push Run Button.\nCaution : please select only one type file (ex : only png gray)')
    self.clear_del_btn()
    self.clear_file_name_labels()

    fileTypes = [("WAV & PNG","*.wav *.png"), ("WAV","*.wav"), ("Gray-Scale PNG","*.png")]
    self.new_file_names_arr = tk.filedialog.askopenfilenames(title='Select file', filetypes=fileTypes, initialdir=self.read_dir)
    if (len(self.new_file_names_arr) > 0):
      self.read_dir = os.path.dirname(self.new_file_names_arr[0])
    
    self.update_file_names_arr()

    self.change_message_label()
    self.make_del_btn()
    self.make_file_name_labels()
  
  # pushed run btn (select save directory of written file)
  def save_dir_dialog(self):
    self.write_dir = tk.filedialog.askdirectory(initialdir=self.write_dir)
  
  # ica for wav
  def ica_wav(self, event):
    self.save_dir_dialog()

    self.process_message_text.set('Processing now ...') #don't reflect this place

    N = len(self.file_names_arr)
    
    data, rates = read_file_wav(self.file_names_arr, self.process_message_text)
    if (self.process_message_text == 'Process suspended by something error.\nCaution : please select only one type file (ex : only png gray)'):
      return
    LEN = len(data[0])
    RATE = rates[0]

    # the data size was different in each wav file(or rate different)
    for i in range(N-1):
      if (len(data[i]) != len(data[i+1])):
        return "error"
      if (rates[i] != rates[i+1]):
        return "error"
    
    tmp_x = []
    for i in range(N):
      tmp_x += make_new_data(data[i],np.mean(data[i]))
    x = np.array(tmp_x).reshape(N,LEN)

    covar_mat = np.cov(x)
    eig_value_data, eig_vector_data = LA.eig(covar_mat)
    D = np.diag(eig_value_data)
    E = eig_vector_data
    V = np.dot(np.dot(E, LA.matrix_power(np.sqrt(D),-1)), E.T)
    z = np.dot(V,x)

    w = []
    for i in range(N):
      w.append(np.array([random.randint(-50,100) for i in range(N)]))
    w_after = []
    for i in range(N):
      w_after.append(repetition(w[i],z))
    
    y = []
    for i in range(N):
      y.append(np.dot(w_after[i].T,z))

    write_file_wav(y, RATE, N, self.write_dir)
    self.process_message_text.set('Process ended successfully!')

  # ica for png(gray)
  def ica_png_gray(self, event):
    self.save_dir_dialog()

    self.process_message_text.set('Processing now ...') #don't reflect this place

    N = len(self.file_names_arr)
    data, images = read_file_png_gray(self.file_names_arr, self.process_message_text)
    if (self.process_message_text == 'Process suspended by something error.\nCaution : please select only one type file (ex : only png gray)'):
      return
    LEN = len(data[0])

    # the data size was different in each png file
    for i in range(N-1):
      if (len(data[i]) != len(data[i+1])):
        return "error"
    
    tmp_x = []
    for i in range(N):
      tmp_x += make_new_data(data[i],np.mean(data[i]))
    x = np.array(tmp_x).reshape(N,LEN)

    covar_mat = np.cov(x)
    eig_value_data, eig_vector_data = LA.eig(covar_mat)
    D = np.diag(eig_value_data)
    E = eig_vector_data
    V = np.dot(np.dot(E, LA.matrix_power(np.sqrt(D),-1)), E.T)
    z = np.dot(V,x)

    w = []
    for i in range(N):
      w.append(np.array([random.randint(-50,100) for i in range(N)]))
    w_after = []
    for i in range(N):
      w_after.append(repetition(w[i],z))
    
    y = []
    for i in range(N):
      y.append(np.dot(w_after[i].T,z))
    
    # convert to 0~255 for png format
    for i in range(N):
      y_max = np.amax(y[i])
      y_min_abs = abs(np.amin(y[i]))
      y[i] += y_min_abs
      y[i] /= y_max + y_min_abs
      y[i] *= 255
    
    # 1d array to 2d array
    images_new = []
    for i in range(N):
      images_new.append(y[i].reshape(images[i].shape[0],images[i].shape[1]))
    
    write_file_png_gray(images_new, N, self.write_dir)
    self.process_message_text.set('Process ended successfully!')

if __name__ == "__main__":
  root = tk.Tk()
  root.title('ICA App')
  root.geometry('800x600')
  app = Application(root)
  app.mainloop()
