# !/usr/bin/env python
# -*- coding: utf-8 -*-

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.plotter import BSPlotter
vasprun = Vasprun("vasprun.xml")
bss = vasprun.get_band_structure(kpoints_filename="KPOINTS", line_mode=True)
plotter = BSPlotter(bss)
#plotter.save_plot("bandStructure.svg", img_format="svg")
#plotter.save_plot("bandStructure.png", img_format="png")
#plotter.save_plot("lim_bandStructure.svg", img_format="svg", ylim=(-.2, 1.4))
plotter.save_plot("MAPbI3-primitive.png", img_format="png", ylim=(-5, 5))
plotter.plot_brillouin()
