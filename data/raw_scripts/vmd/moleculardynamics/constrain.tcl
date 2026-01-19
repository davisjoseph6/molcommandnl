set name ak_open
set old eq1
set oname eq
set stage eq2p
mol new ../constrain/${name}.psf
mol addfile ../${old}/${oname}.restart.coor
set allatoms [atomselect top all]	 
$allatoms set beta 0	 
set fixedatom [atomselect top backbone]
$fixedatom set beta 1
$allatoms set occupancy 0
$allatoms writepdb ../constrain/${stage}/${name}.pdb
exit
