# Kiosk display package for vmd
# Dan Wright <dtwright@uiuc.edu>
# 
# Rotate through molecular scenes using VMD
# Originally written for the door-opening system at the
# UIUC School of Chem. Sciences VizLab
# 
# Usage: source this file in your .vmdrc
# To start the kiosk, call 
#    ::VMDKioskDisplay::init
#    ::VMDKioskDisplay::runKiosk <switch delay in minutes>
#
# To set up the scenes you want to display, add functions
# at the end (where it says "User changable code here")
# and then change the line starting with "set molFuncList"
# in the init function immediately below

namespace eval ::VMDKioskDisplay:: {
	variable molFuncList
	variable funcIdx
	variable switchDelay

	# Used to set up global stuff
	# MUST initialize the function list here!
	proc init { } {
		variable funcIdx
		variable molFuncList
		
		set funcIdx 0
		
		# You need the change this to contain YOUR list
		# of display functions!!
		set molFuncList {1ffy 1r1a 1c0a}
	}

	proc clearVMD { } {
		foreach mol [molinfo list] {
			mol delete $mol
		}
	}

	# Workhorse that runs the molecule switching, etc
	# Cycles between molecule displays defined in 
	# various procs
	#
	# args: delay
	#       Delay between molecules, in minutes
	proc runKiosk { delay } {
		variable molFuncList
		variable funcIdx
		variable switchDelay

		# save delay
		set switchDelay $delay

		# Clear VMD, then invoke the next mol function
		clearVMD
		eval [lindex $molFuncList $funcIdx]

		# and call myself after sleeping the required amount of time
		after [expr $switchDelay*60*1000] { ::VMDKioskDisplay::runKiosk $::VMDKioskDisplay::switchDelay }

		# increment the function index for the next round
		if { $funcIdx < [expr [llength $molFuncList]-1] } {
			incr funcIdx
		} else {
			set funcIdx 0
		}
	}

	# User-changable code here!
	# Start defining molecule switching functions
	# I'm naming them for the pdb codes they load
	proc 1c0a { } {
		set molid [mol new 1c0a]
		
		mol modselect 0 $molid "protein"
		mol modstyle 0 $molid newcartoon
		mol modcolor 0 $molid structure
		mol selection "nucleic"
		mol addrep $molid
		mol modstyle 1 $molid newribbons

		set textid [mol new]
		mol fix $textid
		graphics $textid color black
		graphics $textid text {-0.85 -1.13 0} "E. coli Aspartyl-tRNA Synthetase (1c0a)" size 2

		rock y by 0.25
	}

	proc 1ffy { } {
		set molid [mol new 1ffy]
		
		mol modselect 0 $molid "protein"
		mol modstyle 0 $molid newcartoon
		mol modcolor 0 $molid structure
		mol selection "nucleic"
		mol addrep $molid
		mol modstyle 1 $molid newribbons

		set textid [mol new]
		mol fix $textid
		graphics $textid color black
		graphics $textid text {-0.85 -1.13 0} "Isoleucyl tRNA/tRNA Synthentase (1ffy)" size 2

		rock y by 0.25
	}

	proc 1r1a { } {
		set molid [mol new 1r1a]
		
		mol modselect 0 $molid "protein"
		mol modstyle 0 $molid newcartoon
		mol modcolor 0 $molid structure
		mol selection "not protein"
		mol addrep $molid
		mol modstyle 1 $molid vdw
		mol modcolor 1 $molid type

		set textid [mol new]
		mol fix $textid
		graphics $textid color black
		graphics $textid text {-0.85 -1.13 0} "Human Rhinovirus 1A (1r1a)" size 2

		rock y by 0.25
	}
}
