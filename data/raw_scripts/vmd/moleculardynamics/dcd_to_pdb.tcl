

for { set run 1} {$run < 10} {incr run 1} {
	set prot "GO_3L64"
	set method "Go"
	set segment "1-164"
	#set segment "16-61"
	set infile "../${method}_trajectories/smd100.${segment}_${run}" 

	#mol new ../prot/${prot}.crd
	mol new ../prot/${prot}.psf
	mol addfile ${infile}.dcd type dcd first 0 last  -1 step 1 waitfor all

	set sel [atomselect top all] 
	$sel writepdb ${infile}.pdb
}

exit

