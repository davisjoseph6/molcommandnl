#! /usr/bin/tclsh
#
# vmd_render.tcl -- Render image of PDB and CNS file using VMD
# Copyright (C) 2011 Jeffrey P. Hafner <jphafner@buffalo.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

if { $argc < 4 } {
   puts "USAGE:   vmd -dispdev text -e render.tcl -args <pdbname> <cnsname> <isovalue> <outname>"
   puts "         this script will select CA atoms and based on prot.psf prot.dcd"
   puts "         new protCA.psf protCA.dcd files will be created"
   puts "EXAMPLE: vmd -dispdev text -e render.tcl -args 3eam.pdb 3eam.cns 0.5 output"
   exit
}

set pdbname [lindex $argv 0]
set cnsname [lindex $argv 1]
set isovalue [lindex $argv 2]
set outname [lindex $argv 3]

#display reset
display projection orthographic
#display backgroundgradient off
#display aoambient 0.9
#display aodirect 0.1
#display reposition 0 0
#display distance 10
#display resize 1024 768
#display height 768

axes location Off
#color Display {BackgroundTop} white
#color Display {BackgroundBot} white
color Display Background white

#mol default style tube
#mol default selection protein

#usage: mol <command> [args...]
#
#Molecules and Data:
#  new [file name] [options...]       -- load file into a new molecule
#  new atoms <natoms>                 -- generate a new molecule with 'empty' atoms
#  addfile <file name> [options...]   -- load files into existing molecule
#    options: type, first, last, step, waitfor, volsets, filebonds, autobonds, 
#             molid (addfile only)
#  load <file type> <file name>       -- load molecule (obsolescent)
#  urlload <file type> <URL>          -- load molecule from URL
#  pdbload <four letter accession id> -- download molecule from the PDB
#  cancel <molid>                     -- cancel load/save of trajectory
#  delete <molid>                     -- delete given molecule
#  rename <molid> <name>              -- Rename the specified molecule
#  dataflag <molid> [set | unset] <flagname> -- Set/unset data output flags
#  list [all|<molid>]                 -- displays info about molecules
#
#Molecule GUI Properties:
#  top <molid>                        -- make that molecule 'top'
#  on <molid>                         -- make molecule visible
#  off <molid>                        -- make molecule invisible
#  fix <molid>                        -- don't apply mouse motions
#  free <molid>                       -- let mouse move molecules
#  active <molid>                     -- make molecule active
#  inactive <molid>                   -- make molecule inactive
#
#Graphical Representations:
#  addrep <molid>                     -- add a new representation
#  delrep <repid> <molid>             -- delete rep
#  default [color | style | selection | material] <value>
#  representation|rep <style>         -- set the drawing style for new reps
#  selection <selection>              -- set the selection for new reps
#  color <color>                      -- set the color for new reps
#  material <material>                -- set the material for new reps
#  modstyle <repid> <molid> <style>   -- change the drawing style for a rep
#  modselect <repid> <molid> <selection>  -- change the selection for a rep
#  modcolor <repid> <molid> <color>   -- change the color for a rep
#  modmaterial <repid> <molid> <material> -- change the material for a rep
#  repname <molid> <repid>            -- Get the name of a rep
#  repindex <molid> <repname>         -- Get the repid of a rep from its name
#  reanalyze <molid>                  -- Re-analyze structure after changes
#  bondsrecalc <molid>                -- Recalculate bonds, current timestep
#  ssrecalc <molid>                   -- Recalculate secondary structure (Cartoon)
#  selupdate <repid> <molid> [on|off] -- Get/Set auto recalc of rep selection
#  colupdate <repid> <molid> [on|off] -- Get/Set auto recalc of rep color
#  scaleminmax <molid> <repid> [<min> <max>|auto] -- Get/set colorscale minmax
#  drawframes <molid> <repid> [<framespec>|now] -- Get/Set drawn frame range
#  smoothrep <molid> <repid> [smooth] -- Get or set trajectory smoothing value
#  showperiodic <molid> <repid> [flags] -- Get or set periodic image display
#  numperiodic <molid> <repid> <n>    -- Get or set number of periodic images
#  showrep <molid> <repid> [on|off]   -- Turn selected rep on or off
#
#Clipping Planes:
#  clipplane center <clipid> <repid> <molid> [<vector>]
#  clipplane color  <clipid> <repid> <molid> [<vector>]
#  clipplane normal <clipid> <repid> <molid> [<vector>]
#  clipplane status <clipid> <repid> <molid> [<mode>]
#  clipplane num


## protein structure
mol new $pdbname type pdb
mol delrep 0 top
mol representation Tube
mol color Name
mol selection {all}
mol material Opaque

mol addrep top
mol selupdate 0 top 0
mol colupdate 0 top 0
mol scaleminmax top 0 0.000000 0.000000
mol smoothrep top 0 0
mol drawframes top 0 {now}

## electron density
mol new $cnsname type edm first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
mol delrep 0 top
mol representation Isosurface $isovalue 0 0 0 1 1
mol color Name
mol section {all}
mol material Transparent

mol addrep top
mol selupdate 0 top 0
mol colupdate 0 top 0
mol scaleminmax top 0 0.000000 0.000000
mol smoothrep top 0 0
mol drawframes top 0 {now}

## ART Gelato PostScript Raster3D Radiance Rayshade RenderMan snapshot STL Tachyon TachyonInternal POV3 VRML-1 VRML-2 Wavefront X3D X3DOM
#render Raster3D image.jpg display %s
#render TachyonInternal image
#rend TachyonInternal ${outname}Z.tga {convert $outname.tga $outname.jpg}
#render snapshot MySnapshot convert %s jpg:%s.jpg 
#render PostScript MySnapshot
#render Raster3d render < %s -sgi %s.rgb; ipaste %s.rgb
#render POV3 image.pov

scale by 1.75
rend Tachyon ${outname}.tach

#rend Tachyon image.tach {/util/vmd/v1.9/tachyon/0.99b2/bin/tachyon %s -o output.tga}
## Z Y X
rend TachyonInternal ${outname}Z.tga convert ${outname}Z.tga ${outname}Z.jpg
rotate x by 90 degrees
rend TachyonInternal ${outname}Y.tga convert ${outname}Y.tga ${outname}Y.jpg
rotate y by 270 degrees
rend TachyonInternal ${outname}X.tga convert ${outname}X.tga ${outname}X.jpg

#set outfile [open rmsd.dat w]
#set nf [molinfo top get numframes]
#set frame0 [atomselect top "protein and backbone and noh" frame 0]
#set sel [atomselect top "protein and backbone and noh"]
#set all [atomselect top all]
## rmsd calculation loop
#for { set i 1 } { $i <= $nf } { incr i } {
#    $sel frame $i
#    $all frame $i
#    $all move [measure fit $sel $frame0]
#    puts $outfile "[measure rmsd $sel $frame0]"
#}
#close $outfile

#exit

