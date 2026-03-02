import roboticstoolbox as rtb
import sys

with open("help_output.txt", "w") as f:
    sys.stdout = f
    help(rtb.models.DH.Puma560().plot)
    sys.stdout = sys.__stdout__
