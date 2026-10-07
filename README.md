# tools4magaox
tools and other tricks for processing magAOX data. 


## redu
These programs are intended for getting data in usable order. They include creating the dark, centering both the unsaturated and the coronagraphic images, and filtering these images when creating the final data cube. 

## proc
These programs are for various tests on reduced images. This included setup for calling KLIP routines, calculating contrast curves, etc. Basically, things that don't need to modify the data themselves and reads from the reduced data library. 

## phot 
These scrips are for working with the photometery generated in the proc scripsts. 

## wfs_proc
Wavefront-sensor processing. `loop_reconstruction` loads camwfs xrif frames, reconstructs per-frame delta wavefronts with cacao control matrices, and integrates them into open-loop wavefronts. See [src/tools4magaox/wfs_proc/README.md](src/tools4magaox/wfs_proc/README.md).