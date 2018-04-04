from pymatgen.electronic_structure.plotter import *
from pymatgen.electronic_structure.dos import *
from pymatgen.io.vasp.outputs import *
d = Vasprun('vasprun.xml')
e =BSVasprun('vasprun.xml',parse_projected_eigen=True)
#energy = d.eigenvalues
#ionic = d.ionic_steps
#tdos = d.tdos
cdos = d.complete_dos
bs = e.get_band_structure(line_mode=True)
plotter = BSPlotter(bs)
#c = Dos(d.efermi,d.eigenvalues,d.eigenvalues)
#print tdos##ionic[0]['electronic_steps'][10]
#b = CompleteDos(d.structures,d.)
#a = DosPlotter()
#a.add_dos('cdos',cdos)#get_plot()
#a.show()#save_plot('try')
bd = BSDOSPlotter(tick_fontsize=10,egrid_interval=20,dos_projection="orbitals",bs_legend=None)#bs_projection="HPbCIN",dos_projection="HPbCIN")
bds = bd.get_plot(bs,cdos)
#bds.show()
plotter.plot_brillouin()
#bd._rgbline()
#bb = BSPlotter(bs)
#bb.save_plot('try')
