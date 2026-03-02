import sys
import numpy as np
from PySide6 import QtWidgets
import roboticstoolbox as rtb

def test():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    robot = rtb.models.DH.Puma560()
    # create a dummy trajectory
    q = np.zeros((10, 6))
    for i in range(10):
        q[i, 1] = i * 0.1
    
    # plot and animate
    try:
        print("Starting plot")
        robot.plot(q, backend='pyplot', block=False, dt=0.05)
        print("Plot finished")
    except Exception as e:
        print("Error:", e)
        
    # We won't block the app so it exits
test()
