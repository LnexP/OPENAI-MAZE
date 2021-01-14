# from maze.envs.maze_ import Maze
from tkinter import *
from tkinter import ttk
import time
import threading


class MazeGUI(threading.Thread):
    def __init__(self, obstacles, size) -> None:
        # self.root = Tk()
        # self.canvas = None
        self.gs = 25
        self.w = self.gs * size[0]
        self.h = self.gs * size[1]
        
       
        self.obstacles = obstacles
        # canvas.create_line(0, 1, 800, 1)
        
        threading.Thread.__init__(self)
    
    def run(self):
        self.root = Tk()
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.canvas = Canvas(self.root, width=self.w+50, height=self.h+50)
        self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))
        for i in range(25, self.w+26, self.gs):
            self.canvas.create_line((i, 25, i, self.h+25))
        for i in range(25, self.h+26, self.gs):
            self.canvas.create_line((25, i, self.w+25, i))
        for obs in self.obstacles:
            self.canvas.create_rectangle(25+obs[0]*self.gs, 25+obs[1]*self.gs, 25+obs[0]*self.gs+self.gs, 25+obs[1]*self.gs+self.gs, fill='gray', outline='black')
        self.canvas.create_text(0, 0, text='猫', anchor='center', font='TkMenuFont', tag='cat')
        self.canvas.create_text(0, 0, text='老鼠', anchor='center', font='TkMenuFont', tag='rat')
        self.root.mainloop()








    


    
