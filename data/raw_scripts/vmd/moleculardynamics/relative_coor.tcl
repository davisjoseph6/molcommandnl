#Paul Glenn
# coor.tcl
# Calculate the reaction coordinate progression from pdb file 
set run AK2
set skip 1
set lid 1
set nmp 0
if {0} {
    set domain "not resid 30 to 67 and not resid 118 to 167 and alpha"
    set dname core
}

if {$nmp} {
	set domain "resid 30 to 67 and alpha"
	set dname nmp
    set inclusion "alpha and not resid 118 to 167"
}

if {$lid} {
	set domain "resid 118 to 167 and alpha"
	set dname lid
    set inclusion "not resid 30 to 67 and alpha"
}

set ccr 0
if {$ccr} {
	set dir ../results/end_dry.pdb
	set outfile [open ../analysis/${dname}.dat w]
} else {
	set dir ../protein/iENM_closed.pdb
	#set dir ../results/${run}/end_dry.pdb
	set outfile [open ../analysis/ref_cl/${dname}.dat w]
}

animate read pdb $dir beg 0 end -1 skip $skip waitfor all  
mol new ../protein/4AKE.pdb
mol new ../protein/1AKE.pdb

set selInclusion [ atomselect 0 $inclusion]
set initialInclusion [ atomselect 1 $inclusion]
set finalInclusion [atomselect 2 $inclusion]

set nf [molinfo 0 get numframes]

puts "Domain name: $dname"
puts "Num frames: $nf "
set sel [atomselect 0 $domain]
set init [ atomselect 1 $domain]
set final [atomselect 2 $domain]

for {set i 1} {$i< $nf} {incr i} {
    $sel frame $i
    $selInclusion frame $i

	#Calculate RMSD from initial structure 
    $selInclusion move [measure fit $selInclusion $initialInclusion]
	set RMSDi [measure rmsd $sel $init]
	
	#Calculate RMSD from final structure 
	$selInclusion move [measure fit $selInclusion $finalInclusion]
	set RMSDf [measure rmsd $sel $final] 

    #Calculate reference RMSD from initial to final structure
	$finalInclusion move [measure fit $finalInclusion $initialInclusion]
    set ref_RMSD [measure rmsd $final $init]


    #Calculate the reaction coordinate
	set RC [ expr 0.5*(1 + ($RMSDi*$RMSDi - $RMSDf*$RMSDf)/($ref_RMSD*$ref_RMSD) ) ]
	puts "Reaction coordinate: $RC"
	puts $outfile "$i \t $RC"
}

puts "Domain name: $dname"
exit

