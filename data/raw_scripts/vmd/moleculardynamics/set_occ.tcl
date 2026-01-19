set name ak_f34
mol new ../target_frames/${name}.pdb
set allatoms [atomselect top all]	 	 
set else [atomselect top "not name CA"]	  
$else set occupancy 0
$allatoms writepdb ../target_frames/${name}.pdb
exit
