# Minimal VMD Tcl example
# Usage inside VMD TkConsole:  source load_structure_and_style.tcl
mol new 1crn.pdb
mol delrep 0 top
mol representation NewCartoon
mol color Chain
mol addrep top
display resetview
