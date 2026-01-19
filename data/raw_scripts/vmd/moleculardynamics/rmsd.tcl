
set run_name TMD

animate read pdb ../results/${run_name}/end_dry.pdb beg 0 end -1 skip 1 waitfor all  
set outfile [open ../analysis/$run_name/rmsd.dat w]

set nf [molinfo top get numframes]
set frame0 [atomselect top "protein and alpha" frame 0]
set sel [atomselect top "protein and alpha"]

#RMSD calculation
for {set i 1} {$i< $nf} {incr i} {
    $sel frame $i
    $sel move [measure fit $sel $frame0]
	set rmsd [measure rmsd $sel $frame0]
    puts $rmsd
	puts $outfile $rmsd
}

close $outfile
exit

