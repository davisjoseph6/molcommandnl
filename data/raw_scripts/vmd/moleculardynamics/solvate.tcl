set name ak_f34
set dst ../target_frames/$name
package require solvate
solvate ${dst}.psf ${dst}.pdb -t 5 -o $name
exit
