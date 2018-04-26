from pymatgen.electronic_structure.plotter import DosPlotter,BSPlotter,BSPlotterProjected,BSDOSPlotter
from pymatgen.electronic_structure.dos import Dos,CompleteDos
from pymatgen.electronic_structure.bandstructure import BandStructure,BandStructureSymmLine
from pymatgen.io.vasp.outputs import Vasprun,BSVasprun
from pymatgen.core.structure import Structure
vasp = Vasprun('vasprun.xml')
print(vasp.structures)
bsvasp =BSVasprun('vasprun.xml',parse_projected_eigen=True)
#print("bsvasp:",dir(bsvasp))
structure = Structure.from_file('POSCAR')
#el = structure.composition.elements
#print el
cdos = vasp.complete_dos
#print cdos.get_densities()
tdos = vasp.tdos #Total dos calculated at the end of run
idos = vasp.idos # Integrated dos calculated at the end of run
pdos = vasp.pdos # List of list of PDos objects. Access as pdos[atomindex][orbitalindex]
efermi = vasp.efermi  
eigenvalues = vasp.eigenvalues 
projected_eigenvalues = vasp.projected_eigenvalues
bs = bsvasp.get_band_structure(line_mode=True)
#print(type(bs))
#projection = bs.projections
#get_el_and_orbital = bs.get_projections_on_elements_and_orbitals({'Pb':['s','d']})
#print("el_and_orbital:",get_el_and_orbital)
total_dos = tdos
pdoss = pdos
#CDOS = CompleteDos(structure,total_dos,pdoss)
spd_dos = cdos.get_spd_dos
#print spd_dos
element_dos = cdos.get_element_dos
#element_spd_dos = cdos.get_element_spd_dos(el)
dosplotter = DosPlotter()
Totaldos = dosplotter.add_dos('Total DOS',tdos)
Integrateddos = dosplotter.add_dos('Integrated DOS',idos)
#Pdos = dosplotter.add_dos('Partial DOS',pdos)
#Spd_dos =  dosplotter.add_dos('spd DOS',spd_dos)
#Element_dos = dosplotter.add_dos('Element DOS',element_dos)
#Element_spd_dos = dosplotter.add_dos('Element_spd DOS',element_spd_dos)
dos_dict = {'Total DOS':tdos,'Integrated DOS':idos}#'Partial DOS':pdos,'spd DOS':spd_dos,'Element DOS':element_dos}#'Element_spd DOS':element_spd_dos
add_dos_dict = dosplotter.add_dos_dict(dos_dict)
get_dos_dict = dosplotter.get_dos_dict()
#dos_plot = dosplotter.get_plot()
##dosplotter.save_plot("MAPbI3_dos",img_format="png")
##dos_plot.show()
bsplotter = BSPlotter(bs)
bsplotterprojected = BSPlotterProjected(bs)
bs_plot_data = bsplotter.bs_plot_data()
#bs_plot = bsplotter.get_plot()
#bsplotter.save_plot("MAPbI3_bs",img_format="png")
#bsplotter.show()
ticks = bsplotter.get_ticks()
#print ticks
bsdos = BSDOSPlotter(tick_fontsize=10,egrid_interval=20,legend_fontsize=28,dos_projection="orbitals",bs_legend=None,fig_size=(110, 85))#bs_projection="HPbCIN",dos_projection="HPbCIN")
#projected_plots_dots = bsplotterprojected.get_projected_plots_dots({'Pb':['s','p'],'I':['s','p']})
elements = bsplotterprojected.get_elt_projected_plots(vbm_cbm_marker=True)
bds = bsdos.get_plot(bs,cdos)
bsplotter.plot_brillouin()
