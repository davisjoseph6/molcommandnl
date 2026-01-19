# Paul Glenn
# rg.tcl
# Calculate radius of gyration for frames 

set cp 0
set wt 1

set num_trajectories 25
set segment "16-159"
#set segment "16-61"
set temp "[lindex $argv 0]"
#set trajectory "kp05"
#set trajectory "[lindex $argv 0]"
set method "Go"
if { $wt} {
	set prot "GO_3L64"
	set type "WT"
	set C "resid 60 to 164"
	set N "resid 13 to 59" 
}

if {$cp} {
	set prot "GO_cp_3l64"
	set type "CP"
	set C "resid 60 to 164" 
	set N "resid 13 to 59" 
	
	# In case we realize domains should shift 
	# set N "resid 1 to 47"
	# set C "resid 48 to 164"
}

set summary [open "../summary/${type}/$method.${segment}.txt" w]
puts $summary "C         N"

for { set run 1} {$run <= $num_trajectories} {incr run 1} {
	#set infile "../${method}_trajectories/${trajectory}/smd100.${segment}_${run}.pdb" 
	set infile "../${method}/trajectories/T_${temp}K/T${temp}.${segment}_${run}.pdb"
	
	#mol new ../prot/${prot}.crd
	#mol new ../prot/${prot}.psf
	mol new ../prot/${prot}.pdb
	mol addfile ${infile} type pdb first 0 last  -1 step 1 waitfor all
	set C_domain [atomselect top "resid 60 to 164 and name CA"]
	set N_domain [atomselect top "resid 13 to 59 and name CA"]

	set nf [molinfo top get numframes] 

	set outfileC [open "../analysis/${type}/${method}.${segment}_${run}_RG_C.txt" w]
	set outfileN [open "../analysis/${type}/${method}.${segment}_${run}_RG_N.txt" w]
	puts "Num frames: $nf "

	proc gyr_radius {sel} {
	  # make sure this is a proper selection and has atoms
	  if {[$sel num] <= 0} {
		error "gyr_radius: must have at least one atom in selection"
	  }
	  # gyration is sqrt( sum((r(i) - r(center_of_mass))^2) / N)
	  set com [measure center $sel weight mass]
	  set sum 0
	  foreach coord [$sel get {x y z}] {
		set sum [vecadd $sum [veclength2 [vecsub $coord $com]]]
	  }
	  return [expr sqrt($sum / ([$sel num] + 0.0))]
	}

	for {set i 1} {$i< $nf} {incr i 1} {
		$C_domain frame $i
		$N_domain frame $i
			
		#set comC  [measure center $C_domain weight mass]
		#set comN  [measure center $N_domain weight mass] 
		
		set RG_C [gyr_radius ${C_domain}]
		set RG_N [gyr_radius ${N_domain}]

		#Normalize 
		set RG_C [expr $RG_C / 13.61 ]
		set RG_N [expr $RG_N / 9.68 ] 
		puts "$i $RG_C $RG_N"
		puts $outfileC $RG_C
		puts $outfileN $RG_N
	}

	mol delete all 
	puts $summary "$RG_C $RG_N"
	close $outfileC
	close $outfileN
	
}
#puts "Completed: $trajectory"
close $summary
exit
