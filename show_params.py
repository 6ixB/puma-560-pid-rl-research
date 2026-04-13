import roboticstoolbox as rtb
import numpy as np

def show_puma_parameters():
    # Load the PUMA 560 model
    robot = rtb.models.DH.Puma560()
    
    print("=" * 60)
    print(f"{'Link':<5} | {'Mass (kg)':<10} | {'Motor J (kgm2)':<15} | {'Gear Ratio':<12} | {'Friction (B)':<10}")
    print("-" * 60)
    
    for i, link in enumerate(robot.links):
        # Handle cases where attributes might be None or need special formatting
        mass = link.m if link.m is not None else 0.0
        jm = link.Jm if link.Jm is not None else 0.0
        g = link.G if link.G is not None else 1.0
        b = link.B if link.B is not None else 0.0
        
        print(f"{i:<5} | {mass:<10.4f} | {jm:<15.4f} | {g:<12.4f} | {b:<10.4f}")

    print("=" * 60)
    print("\nExample: Detailed View for Link 2 (Shoulder)")
    link2 = robot.links

    for (i, link) in enumerate(link2):
        print("Link " + str(i))
        print(f"Mass (m): {link.m}")
        print(f"Center of Mass (r):\n{link.r}")
        print(f"Inertia Tensor (I):\n{link.I}")
        print(f"Gear Ratio (G): {link.G}")
        print(f"Viscous Friction (B): {link.B}")
        print(f"Coulomb Friction (Tc): {link.Tc}")
        print("\n")

if __name__ == "__main__":
    show_puma_parameters()
