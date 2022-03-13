import tkinter as tk
import tkinter.filedialog

class Application(tk.Frame):
  def __init__(self, root=None):
    super().__init__(root, width=800, height=600)
    self.root = root
    self.pack()

    self.create_fileopen_btn()
    self.create_exe_btn()

    self.file_name = tk.StringVar()
    self.file_name.set('File unselected')

    self.create_label()

  def create_fileopen_btn(self):
    fileopen_btn = tk.Button(self, text='Select *.wav file', bg="#ffffff", activebackground="#a9a9a9",relief='raised')
    fileopen_btn.bind('<ButtonPress>', self.file_dialog)
    fileopen_btn.place(x=20, y=20, width=200, height=20)
  
  def create_exe_btn(self):
    exe_btn = tk.Button(self, text='Run', bg="#ffffff", activebackground="#a9a9a9")
    #exe_btn.bind('<ButtonPress>', self.file_dialog)
    exe_btn.place(x=240, y=20, width=200, height=20)

  def create_label(self):
    label = tk.Label(textvariable=self.file_name)
    label.place(x=20, y=40, width=700, height=20)
  
  def file_dialog(self, event):
    fileTypes = [("WAV","*.wav")]
    initialDir = "./"
    file_name = tk.filedialog.askopenfilename(title='Select file', filetypes=fileTypes, initialdir=initialDir)
    if (len(file_name)==0):
      self.file_name.set('File unselected')
    else:
      self.file_name.set(file_name)


if __name__ == "__main__":
  root = tk.Tk()
  root.title('ICA App')
  root.geometry('800x600')
  app = Application(root)
  app.mainloop()
