set name TMDk300
set pdbFile ../results/$name/end_dry.pdb
set traj_psf    ../protein/ak_open.psf
set traj_dry ../results/$name/end_dry.dcd

mol new ${traj_psf} type psf
animate read pdb  $pdbFile beg 0 end -1 skip 1 waitfor all
animate write dcd ${traj_dry} waitfor all sel [atomselect top all] 
exit
