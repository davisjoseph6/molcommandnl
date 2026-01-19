# convert dcds to "dry" pdb frames 
set traj_psf    ../protein/ak_open.psf
set traj_dcd0  /panasas/scratch/paulglen/ADK/tmd01/tmd.dcd
set traj_dcd1  /panasas/scratch/paulglen/ADK/eqtmd/tmd01/eq.dcd
set traj_dcd2  /panasas/scratch/paulglen/ADK/tmd02/tmd.dcd
set traj_dcd3  /panasas/scratch/paulglen/ADK/eqtmd/tmd02/eq.dcd
set traj_dcd4  /panasas/scratch/paulglen/ADK/tmd05/tmd.dcd
set traj_dcd5  /panasas/scratch/paulglen/ADK/eqtmd/tmd05/tmd.dcd
set traj_dcd6  /panasas/scratch/paulglen/ADK/tmd11/tmd.dcd
set traj_dcd7  /panasas/scratch/paulglen/ADK/eqtmd/tmd11/tmd.dcd
set traj_dcd8  /panasas/scratch/paulglen/ADK/tmd21/tmd.dcd
set traj_dcd9  /panasas/scratch/paulglen/ADK/eqtmd/tmd21/tmd.dcd
set traj_dcd10 /panasas/scratch/paulglen/ADK/tmd34/tmd.dcd
set traj_dcd11 /panasas/scratch/paulglen/ADK/eqtmd/tmd34/tmd.dcd

set traj_dry  ../results/end_dry.pdb

mol new ${traj_psf} type psf

animate read dcd ${traj_dcd0} beg 0 end -1 skip 10 waitfor all  
animate read dcd ${traj_dcd1} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd2} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd3} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd4} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd5} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd6} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd7} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd8} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd9} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd10} beg 0 end -1 skip 10 waitfor all
animate read dcd ${traj_dcd11} beg 0 end -1 skip 10 waitfor all

animate write pdb ${traj_dry} waitfor all sel [atomselect top "not water"] 

exit
