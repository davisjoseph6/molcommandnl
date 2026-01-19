#Paul Glenn
# reaction.tcl
#Calculate the reaction coordinate progression from pdb file 
set skip 1 
set run_name AK2
animate read pdb ../results/${run_name}/end_dry.pdb beg 0 end -1 skip $skip waitfor all  
mol new ../protein/4AKE.pdb
mol new ../protein/1AKE.pdb

set outfile [open ../analysis/${run_name}/coor.dat w]
set init [ atomselect 1 "name CA"]
set final [atomselect 2 "name CA"]
set sel [atomselect 0 "name CA"]
set frame0 [atomselect top "name CA" frame 0]
set nf [molinfo 0 get numframes]

puts "Num frames: $nf "
for {set i 1} {$i< $nf} {incr i} {
    $sel frame $i

	#Calculate RMSD from initial structure 
    $sel move [measure fit $sel $init]
	set RMSDi [measure rmsd $sel $init]
	
	#Calculate RMSD from final structure 
	$sel move [measure fit $sel $final]
	set RMSDf [measure rmsd $sel $final] 

    #Calculate reference RMSD from initial to final structure
	$final move [measure fit $final $init]
    set ref_RMSD [measure rmsd $final $init]

    #Calculate the reaction coordinate
	set RC [ expr 0.5*(1 + ($RMSDi*$RMSDi - $RMSDf*$RMSDf)/($ref_RMSD*$ref_RMSD) ) ]
	puts "Reaction coordinate: $RC"
	puts $outfile "$i \t $RC"
}

exit

