set frameSet {01 02 05 11 21 33}
set class apo

#set frameID
animate read pdb ../protein/iENM_open.pdb beg 1 end -1 waitfor all


foreach frameID $frameSet {
	set sel [atomselect top  all frame $frameID]
	$sel writepdb ../protein/${class}/f_${frameID}.pdb
}

exit
