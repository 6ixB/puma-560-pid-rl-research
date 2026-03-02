import roboticstoolbox as rtb
import numpy as np

robot = rtb.models.DH.Puma560()
robot.teach(q=[0, 0, 0, 0, 0, 0], backend='pyplot')

print(robot.qlim)

limits_deg = robot.qlim * (180.0/np.pi)
print(limits_deg)

# q1 -> -160 - 160
# q2 -> -110 - 110