import tkinter as tk
import tkinter.filedialog
import numpy as np
import numpy.linalg as LA
import random
import datetime
from scipy.io.wavfile import read, write


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
def read_file_wav(filenames):
  rates = []
  data = []
  for f in filenames:
    tmp_rate, tmp_data = read(f)
    rates.append(tmp_rate)
    data.append(tmp_data)
  return data, rates

# write data(wav)
def write_file_wav(y, RATE, N):
  date_now = datetime.datetime.now().isoformat()
  s = []
  for i in range(N):
    s.append('separate'+date_now+'_'+str(N+1)+'.wav')
  for i in range(N):
    write(s[i], RATE, y[i])

# ica for wav
def ica_wav(filenames):
  N = len(filenames)
  data, rates = read_file_wav(filenames)
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
    w.append([random.random() for i in range(N)])
  w_after = []
  for i in range(N):
    w_after.append(repetition(w[i],z))
  
  y = []
  for i in range(N):
    y.append(np.dot(w_after[i].T,z))
  
  write_file_wav(y, RATE, N)


class Application(tk.Frame):
  def __init__(self, root=None):
    super().__init__(root, width=800, height=600)
    self.root = root
    self.pack()

    #file names array
    self.new_file_names_arr = [] # file names added by select btn
    self.file_names_arr = [] # file names to sep

    #file open btn
    fileopen_btn = tk.Button(self, text='Select *.wav file', bg="#ffffff", activebackground="#a9a9a9", relief='raised')
    fileopen_btn.bind('<ButtonPress>', self.file_dialog)
    fileopen_btn.place(x=20, y=20, width=200, height=20)

    #run ICA btn
    self.exe_btn = tk.Button(self, text='Run', bg="#ffffff", activebackground="#a9a9a9", relief='raised', state=tk.DISABLED)
    #self.exe_btn.bind('<ButtonPress>', ica_wav(self.file_names_arr))
    self.exe_btn.place(x=240, y=20, width=200, height=20)

    # message
    self.message = 'File unselected'
    self.message_label = tk.Label(text=self.message)
    self.message_label.place(x=40, y=40, width=700, height=20)

    # del_btns
    self.del_btns = []

    #file name labels
    self.file_name_labels = []
  
  # make file name labels
  def make_file_name_labels(self):
    for i in range(len(self.file_names_arr)):
      self.file_name_labels.append(tk.Label(text=self.file_names_arr[i], anchor='w'))
      self.file_name_labels[i].place(x=40, y=60+20*i, width=700, height=20)
  
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
      self.exe_btn["state"] = tk.DISABLED
    else:
      self.message = 'If you want to delete file, please click the number button on the left side of file name'
      if (len(self.file_names_arr)==1):
        self.exe_btn["state"] = tk.DISABLED
      else:
        self.exe_btn["state"] = tk.NORMAL
    self.message_label['text'] = self.message

  # pushed select file btn
  def file_dialog(self, event):
    self.clear_del_btn()
    self.clear_file_name_labels()

    fileTypes = [("WAV","*.wav")]
    initialDir = "./"
    self.new_file_names_arr = tk.filedialog.askopenfilenames(title='Select file', filetypes=fileTypes, initialdir=initialDir)
    
    self.update_file_names_arr()

    self.change_message_label()
    self.make_del_btn()
    self.make_file_name_labels()
  
  #def exe_ica(self, event):

if __name__ == "__main__":
  root = tk.Tk()
  root.title('ICA App')
  root.geometry('800x600')
  app = Application(root)
  app.mainloop()
