set name path_AKPG
animate read pdb ../protein/${name}.pdb beg 0 end -1 skip 1 waitfor all  
set nf [molinfo top get numframes]
set frame0 [atomselect top "name CA" frame 0]
set sel [atomselect top "name CA"]
set sel2 [atomselect top "name CA"]
for {set j 1} {$j < $nf} {incr j} { 
$sel2 frame $nf
	for {set i 1} {$i< $nf} {incr i} {
		$sel frame $i
		$sel move [measure fit $sel $frame0]
		puts " $i \t [measure rmsd $sel $frame0]"
		
	}

}
exit
