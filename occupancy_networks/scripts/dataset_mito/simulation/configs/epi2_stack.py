'''
 # @ Author: Abhinanda Ranjit Punnakkal
 # @ Create Time: 2023-08-16 13:42:55
 # @ Modified by: Abhinanda Ranjit Punnakkal
 # @ Modified time: 2023-08-16 13:55:59
 # @ Description: Configuration for Epi1Mito Arif paper
 '''

from types import SimpleNamespace
import math as m

# Microscope PSF parameters.
m_params = {"M" : 60,              # magnification
            "NA" : 1.4,              # numerical aperture
            "ng0" : 1.515,           # coverslip RI design value
            "ng" : 1.515,            # coverslip RI experimental value
            "ni0" : 1.515,           # immersion medium RI design value
            "ni" : 1.515,            # immersion medium RI experimental value
            "ns" : 1.33,             # specimen refractive index (RI)
            "ti0" : 150,             # microns, working distance (immersion medium thickness) design value
            "tg" : 170,              # microns, coverslip thickness experimental value
            "tg0" : 170,             # microns, coverslip thickness design value
            "zd0" : 200.0 * 1.0e+3}  # microscope tube length (in microns).

sim_params = SimpleNamespace(
    # Microscope para
    step_size_xy =  0.080,      # [um] pixel size
    wvl = .608,                   # nm

    #Noise parameters
    n_low_min = 20,
    n_low_max = 70,
    n_high_min = 200,
    n_high_max = 240,

    # Output parameters
    size_x = 128,               # size of x/y in pixels (image)
    size_t = 1,                 # number of acquisitions

    # Views and slicing
    angles= [0 , m.pi, m.pi/2.],  # 

    axes = ['x','y', 'z', ],
    z_transes = [ 0, -0.25, -0.5],


)