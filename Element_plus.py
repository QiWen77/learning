from pymatgen.core.periodic_table import Element

def list_filter_elements(filter_function=None):

    """
    A pretty ASCII printer for the periodic table, based on some
    filter_function.

    Args:
        filter_function: A filtering function taking an Element as input
            and returning a boolean. For example, setting
            filter_function = lambda el: el.X > 2 will print a periodic
            table containing only elements with electronegativity > 2.
    """

    filter_els = []
    for atomic_no in range(1, 103):
        try:
            el = Element.from_Z(atomic_no)
        except ValueError:
            el = None
        if el and ((not filter_function) or filter_function(el)):
            filter_els.append("{:s}".format(el.symbol))
    return filter_els

