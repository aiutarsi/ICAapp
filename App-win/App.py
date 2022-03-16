import tkinter as tk
import tkinter.filedialog
import numpy as np
import numpy.linalg as LA
import random
import datetime
from scipy.io.wavfile import read, write
from PIL import Image, ImageOps
import os
import copy

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
      if (abs(abs(w_before[i]) - abs(w[i]))<0.000000001):
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
      return "error", "error"
  return data, rates

# read data from file(png gray)
def read_file_png_gray(filenames, process_message_text):
  images = []
  for f in filenames:
    try:
      images.append(np.array(Image.open(f).convert('L')))
    except:
      process_message_text.set('Process suspended by something error.\nCaution : please select only one type file (ex : only png gray)')
      return "error", "error"
  data = []
  for i in range(len(filenames)):
    data.append(images[i].copy())
  for i in range(len(filenames)):
    data[i] = np.ravel(data[i])
  return data, images


def three_to_one_color(im, height, width, num):
  array = []
  im_cp = np.array(im)
  for i in range(height):
    for j in range(width):
      array.append(im_cp[i][j][num])
  return array

def one_to_three_color(array, height, width, num, img):
  count = 0
  for i in range(height):
    for j in range(width):
      img[i][j][num] = array[count]
      count += 1

# read data from file(png color)
def read_file_png_color(filenames, process_message_text):
  images = []
  for f in filenames:
    try:
      images.append(np.array(Image.open(f)))
    except:
      process_message_text.set('Process suspended by something error.\nCaution : please select only one type file (ex : only png gray)')
      return "error", "error"

  images_copy = []
  for i in range(len(filenames)):
    images_copy.append(images[i])
  
  for i in range(len(filenames)-1):
    if (images[i].shape != images[i+1].shape):
      process_message_text.set('the size of png(color) file is different each other.')
      return "error", "error"
  
  if (len(filenames) <= 1):
    process_message_text.set('no selected file.')
    return "error", "error"
  if (len(images[0].shape) != 3):
    process_message_text.set('some files is not format: png color.')
    return "error", "error"

  height = images[0].shape[0]
  width = images[0].shape[1]
  color = images[0].shape[2]
  if (color < 3):
    process_message_text.set('some files is not rgb color.')
    return "error", "error"
  
  # 3d array -> 1d array
  data = []
  for i in range(len(filenames)):
    data.append([])
    for j in range(color):
      data[i].append(np.array(three_to_one_color(images_copy[i],height,width,j)))
  
  return data, images


# write data(wav)
def write_file_wav(y, RATE, N, write_dir):
  date_now = datetime.datetime.now().isoformat()
  date_now = date_now.replace(':', '-')
  s = []
  for i in range(N):
    s.append(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'.wav')
    #s.append('separate_'+date_now+'_'+str(i+1)+'.wav')
  for i in range(N):
    write(s[i], RATE, y[i])

# write data(png gray)
def write_file_png_gray(images_new, N, write_dir):
  images = []
  for i in range(N):
    images.append(Image.fromarray(images_new[i].astype(np.uint8)))
  date_now = datetime.datetime.now().isoformat()
  date_now = date_now.replace(':', '-')
  s = []
  for i in range(N):
    s.append(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'.png')
    #s.append('separate_'+date_now+'_'+str(i+1)+'.png')
  for i in range(N):
    images[i].save(s[i])
  # convert black & white
  for i in range(N):
    image_inv = ImageOps.invert(images[i])
    image_inv.save(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'_inv.png')
    #image_inv.save('separate_'+date_now+'_'+str(i+1)+'_inv.png')

# write data(png color)
def write_file_png_color(images_each_color_sep_tmp, images_each_color_sep, N, write_dir, height, width):
  date_now = datetime.datetime.now().isoformat()
  date_now = date_now.replace(':', '-')
  for i in range(N):
    images_each_color_sep[i][0].save(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'_red.png')
    images_each_color_sep[i][1].save(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'_green.png')
    images_each_color_sep[i][2].save(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'_blue.png')

  images_each_color_sep_tmp_inv = copy.deepcopy(images_each_color_sep_tmp)
  for i in range(N):
    for j in range(3):
      for h in range(height):
        for w in range(width):
          images_each_color_sep_tmp_inv[i][j][h][w][j] = 255 - images_each_color_sep_tmp_inv[i][j][h][w][j]
  
  images_each_color_sep_inv = [[] for i in range(N)]
  for i in range(N):
    for j in range(3):
      images_each_color_sep_inv[i].append([])
  for i in range(N):
    for j in range(3):
      images_each_color_sep_inv[i][j] = Image.fromarray(images_each_color_sep_tmp_inv[i][j].astype(np.uint8))
  for i in range(N):
    images_each_color_sep_inv[i][0].save(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'_red_inv.png')
    images_each_color_sep_inv[i][1].save(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'_green_inv.png')
    images_each_color_sep_inv[i][2].save(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'_blue_inv.png')

  images_sep_tmp = []
  images_sep = []
  for i in range(N):
    images_sep_tmp.append(np.zeros((height,width,3)))
  for h in range(N):
    for i in range(height):
      for j in range(width):
        for k in range(3):
          images_sep_tmp[h][i][j][k] = images_each_color_sep_tmp[h][k][i][j][k]
  for i in range(N):
    images_sep.append(Image.fromarray(images_sep_tmp[i].astype(np.uint8)))
  for i in range(N):
    images_sep[i].save(write_dir+os.sep+'separate_'+date_now+'_'+str(i+1)+'.png')
    #images_sep[i].save('separate_'+date_now+'_'+str(i+1)+'.png')


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
    fileopen_btn.place(x=20, y=20, width=120, height=20)

    #run ICA btn (wav)
    self.exe_btn_wav = tk.Button(self, text='Run (wav)', bg="#ffffff", activebackground="#a9a9a9", relief='raised', state=tk.DISABLED)
    self.exe_btn_wav.bind('<ButtonPress>', self.ica_wav)
    self.exe_btn_wav.place(x=150, y=20, width=120, height=20)

    #run ICA btn (png gray)
    self.exe_btn_png_g = tk.Button(self, text='Run (png gray)', bg="#ffffff", activebackground="#a9a9a9", relief='raised', state=tk.DISABLED)
    self.exe_btn_png_g.bind('<ButtonPress>', self.ica_png_gray)
    self.exe_btn_png_g.place(x=280, y=20, width=120, height=20)

    #run ICA btn (png color)
    self.exe_btn_png_c = tk.Button(self, text='Run (png color)', bg="#ffffff", activebackground="#a9a9a9", relief='raised', state=tk.DISABLED)
    self.exe_btn_png_c.bind('<ButtonPress>', self.ica_png_color)
    self.exe_btn_png_c.place(x=410, y=20, width=120, height=20)

    #run synthesize btn
    self.exe_btn_syn = tk.Button(self, text='synthesize png(r,g,b)', bg="#ffffff", activebackground="#a9a9a9", relief='raised', state=tk.DISABLED)
    self.exe_btn_syn.bind('<ButtonPress>', self.synthesize_rgb)
    self.exe_btn_syn.place(x=540, y=20, width=140, height=20)

    # message
    self.message = 'File unselected'
    self.message_label = tk.Label(text=self.message)
    self.message_label.place(x=40, y=40, width=700, height=20)

    # del_btns
    self.del_btns = []

    # file name labels
    self.file_name_labels = []

    # read data & save data directory
    self.read_dir = os.getcwd()
    self.write_dir = os.getcwd()

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
      self.exe_btn_png_c["state"] = tk.DISABLED
      self.exe_btn_syn["state"] = tk.DISABLED
    else:
      self.message = 'If you want to delete file, please click the number button on the left side of file name'
      if (len(self.file_names_arr)==1):
        self.exe_btn_wav["state"] = tk.DISABLED
        self.exe_btn_png_g["state"] = tk.DISABLED
        self.exe_btn_png_c["state"] = tk.DISABLED
        self.exe_btn_png_c["state"] = tk.DISABLED
      else:
        self.exe_btn_wav["state"] = tk.NORMAL
        self.exe_btn_png_g["state"] = tk.NORMAL
        self.exe_btn_png_c["state"] = tk.NORMAL
        if (len(self.file_names_arr)==3):
          self.exe_btn_syn["state"] = tk.NORMAL
        else:
          self.exe_btn_syn["state"] = tk.DISABLED
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
    self.write_dir = self.write_dir.replace('/', os.sep)
    #print(self.write_dir)
  
  # ica for wav
  def ica_wav(self, event):
    self.save_dir_dialog()

    self.process_message_text.set('Processing now ...(may be some error)') #don't reflect this place

    N = len(self.file_names_arr)
    
    data, rates = read_file_wav(self.file_names_arr, self.process_message_text)
    if (self.process_message_text == 'Process suspended by something error.\nCaution : please select only one type file (ex : only png gray)'):
      return
    LEN = len(data[0])
    RATE = rates[0]

    # the data size was different in each wav file(or rate different)
    for i in range(N-1):
      if (len(data[i]) != len(data[i+1])):
        self.process_message_text.set('the size of wav file is different each other.')
        return
      if (rates[i] != rates[i+1]):
        self.process_message_text.set('the sampling frequency of wav file is different each other.')
        return
    
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

    w_after = []
    for i in range(N):
      w = np.random.rand(N)
      norm_w = LA.norm(w)
      for i2 in range(N):
        w[i2] /= norm_w
      w_after.append(repetition(w,z))
    
    y = []
    for i in range(N):
      y.append(np.dot(w_after[i].T,z))

    write_file_wav(y, RATE, N, self.write_dir)
    self.process_message_text.set('Process ended successfully!')

  # ica for png(gray)
  def ica_png_gray(self, event):
    self.save_dir_dialog()

    self.process_message_text.set('Processing now ...(may be some error)') #don't reflect this place

    N = len(self.file_names_arr)
    data, images = read_file_png_gray(self.file_names_arr, self.process_message_text)
    if (self.process_message_text == 'Process suspended by something error.\nCaution : please select only one type file (ex : only png gray)'):
      return
    LEN = len(data[0])

    # the data size was different in each png file
    for i in range(N-1):
      if (len(data[i]) != len(data[i+1])):
        self.process_message_text.set('the size of png(gray) file is different each other.')
        return
    
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

    w_after = []
    for i in range(N):
      w = np.random.rand(N)
      norm_w = LA.norm(w)
      for i2 in range(N):
        w[i2] /= norm_w
      w_after.append(repetition(w,z))
    
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
  
  # ica for png(color)
  def ica_png_color(self, event):
    self.save_dir_dialog()

    self.process_message_text.set('Processing now ...(may be some error)') #don't reflect this place

    N = len(self.file_names_arr)
    data, images = read_file_png_color(self.file_names_arr, self.process_message_text)
    if (data == "error"):
      return
    
    height = images[0].shape[0]
    width = images[0].shape[1]
    color = images[0].shape[2]

    zeros = np.zeros((height*width))

    # mix png in each color(r:0, g:1, b:2)
    images_each_color_mix = [[] for i in range(N)] #[file num][color]
    images_each_color_mix_tmp = [[] for i in range(N)]
    for i in range(N):
      for j in range(3):
        images_each_color_mix[i].append([])
        images_each_color_mix_tmp[i].append([])

    for i in range(N):
      for j in range(3):
        images_each_color_mix_tmp[i][j] = np.zeros((height,width,3))
        for k in range(3):
          if (j == k):
            one_to_three_color(data[i][j],height,width,k,images_each_color_mix_tmp[i][j])
          else:
            one_to_three_color(zeros,height,width,k,images_each_color_mix_tmp[i][j])
        images_each_color_mix[i][j] = Image.fromarray(images_each_color_mix_tmp[i][j].astype(np.uint8))
    
    # ica in each color
    images_each_color_sep = [[] for i in range(N)] #[file num][color]
    images_each_color_sep_tmp = [[] for i in range(N)]
    for i in range(N):
      for j in range(3):
        images_each_color_sep[i].append([])
        images_each_color_sep_tmp[i].append([])

    for i in range(3):
      data_each = []
      for j in range(N):
        data_each.append(np.array(three_to_one_color(images_each_color_mix[j][i],height,width,i)))
      LEN = len(data_each[0])

      tmp_x = []
      for j in range(N):
        tmp_x += make_new_data(data_each[j],np.mean(data_each[j]))
      x = np.array(tmp_x).reshape(N,LEN)

      covar_mat = np.cov(x)
      eig_value_data, eig_vector_data = LA.eig(covar_mat)
      D = np.diag(eig_value_data)
      E = eig_vector_data
      V = np.dot(np.dot(E, LA.matrix_power(np.sqrt(D),-1)), E.T)
      z = np.dot(V,x)

      w_after = []
      for j in range(N):
        w = np.random.rand(N)
        norm_w = LA.norm(w)
        for i2 in range(N):
          w[i2] /= norm_w
        w_after.append(repetition(w,z))
    
      y = []
      for j in range(N):
        y.append(np.dot(w_after[j].T,z))
      
      # convert to 0~255 for png format
      for j in range(N):
        y_max = np.amax(y[j])
        y_min_abs = abs(np.amin(y[j]))
        y[j] += y_min_abs
        y[j] /= y_max + y_min_abs
        y[j] *= 255
      
      # 1d array -> 3d array
      zeros = np.zeros((height*width))
      for j in range(N):
        images_each_color_sep_tmp[j][i] = np.zeros((height,width,3))
        for k in range(3):
          if (i == k):
            one_to_three_color(y[j],height,width,k,images_each_color_sep_tmp[j][i])
          else:
            one_to_three_color(zeros,height,width,k,images_each_color_sep_tmp[j][i])
        images_each_color_sep[j][i] = Image.fromarray(images_each_color_sep_tmp[j][i].astype(np.uint8))

    write_file_png_color(images_each_color_sep_tmp,images_each_color_sep, N, self.write_dir, height, width)
    self.process_message_text.set('Process ended successfully!')

  # synthesize
  def synthesize_rgb(self, event):
    self.save_dir_dialog()

    self.process_message_text.set('Processing now ...(may be some error)') #don't reflect this place

    N = len(self.file_names_arr)
    images = []
    for f in self.file_names_arr:
      try:
        images.append(np.array(Image.open(f)))
      except:
        self.process_message_text.set('Process suspended by something error.\nCaution : please select only one type file (ex : only png gray)')
        return

    images_copy = []
    for i in range(len(self.file_names_arr)):
      images_copy.append(images[i])
  
    for i in range(len(self.file_names_arr)-1):
      if (images[i].shape != images[i+1].shape):
        self.process_message_text.set('the size of png(color) file is different each other.')
        return
  
    if (len(self.file_names_arr) != 3):
      self.process_message_text.set('Please select only 3 files :png (red, green, blue).')
      return
    if (len(images[0].shape) != 3):
      self.process_message_text.set('some files is not format: png color.')
      return

    height = images[0].shape[0]
    width = images[0].shape[1]
    color = images[0].shape[2]
    if (color < 3):
      self.process_message_text.set('some files is not rgb color.')
      return
    
    image_new_arr = np.zeros((height,width,3))
    for i in range(N):
      c = 0
      flag = False
      for j in range(3):
        for k in range(height):
          for l in range(width):
            if (images[i][k][l][j] != 0):
              c = j
              flag = True
              break
          if(flag): break
        if (flag): break
      for h in range(height):
        for w in range(width):
          image_new_arr[h][w][c] = images[i][h][w][c]
    
    image_new = Image.fromarray(image_new_arr.astype(np.uint8))
    date_now = datetime.datetime.now().isoformat()
    date_now = date_now.replace(':', '-')
    image_new.save(self.write_dir+os.sep+'synthesize_'+date_now+'.png')
    #image_new.save('synthesize_'+date_now+'.png')
    self.process_message_text.set('Process ended successfully!')


if __name__ == "__main__":
  root = tk.Tk()
  root.title('ICA App')
  root.geometry('800x600')
  app = Application(root)
  app.mainloop()
