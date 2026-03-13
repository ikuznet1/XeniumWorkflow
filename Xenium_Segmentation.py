#!/usr/bin/env python

########################################################
########################################################
### Extracting images
########################################################
########################################################

import tifffile

# Variable 'LEVEL' determines the level to extract. It ranges from 0 (highest
# resolution) to 6 (lowest resolution) for morphology.ome.tif
LEVEL = 2

with tifffile.TiffFile('morphology.ome.tif') as tif:
    image = tif.series[0].levels[LEVEL].asarray()

tifffile.imwrite(
    'level_'+str(LEVEL)+'_morphology.ome.tif',
    image,
    photometric='minisblack',
    dtype='uint16',
    tile=(1024, 1024),
    compression='JPEG_2000_LOSSY',
    metadata={'axes': 'ZYX'},
)


########################################################
########################################################
### Image info
########################################################
########################################################

#!/usr/bin/env python

import tifffile

with tifffile.TiffFile('morphology.ome.tif') as tif:
    for tag in tif.pages[0].tags.values():
        if tag.name == "ImageDescription":
            print(tag.name+":", tag.value)

#Level 0 0.2125
#Level 1 0.425
#Level 2 0.85
#Level 3 1.7
#Level 4 3.4
#Level 5 6.8
#Level 6 13.6

########################################################
########################################################
### Run Cellpose
########################################################
########################################################

python -m cellpose --dir ./Cellpose --pretrained_model nuclei --chan 0 --chan2 0 --img_filter _morphology.ome --diameter 6.8 --do_3D --save_tif --verbose


########################################################
########################################################
### Proseg
########################################################
########################################################

proseg --xenium ./20241206_KristinaRawData/output-XETG00217__0038213__Region_1__20241206__182124/transcripts.parquet
proseg --xenium ./20241206_KristinaRawData/output-XETG00217__0038216__Region_1__20241206__182124/transcripts.parquet 
proseg --xenium ./20241212_KristinaRawData/output-XETG00217__0038290__Region_1__20241212__142808/transcripts.parquet 
proseg --xenium ./20241212_KristinaRawData/output-XETG00217__0038290__Region_2__20241212__142808/transcripts.parquet 
proseg --xenium ./20241212_KristinaRawData/output-XETG00217__0038291__Region_1__20241212__142808/transcripts.parquet 
proseg --xenium ./20241212_KristinaRawData/output-XETG00217__0038291__Region_2__20241212__142808/transcripts.parquet 



/home/kli1/.cargo/bin/proseg-to-baysor transcript-metadata.csv.gz cell-polygons.geojson.gz \
    --output-transcript-metadata baysor-transcript-metadata.csv \
    --output-cell-polygons baysor-cell-polygons.geojson


#docker run --rm -ti --platform linux/amd64 --privileged=true -v '/Volumes/Extreme SSD/Xenium':/files ubuntu:latest
#apt-get update && apt-get install -y wget
#wget -O xeniumranger-3.1.0.tar.gz "https://cf.10xgenomics.com/releases/xeniumranger/xeniumranger-3.1.0.tar.gz?Expires=1735612672&Key-Pair-Id=APKAI7S6A5RYOXBWRPDA&Signature=M0Ojt5iX42ONOUoaek781oVLkMOdJkwyWWTr9OoL-P~hPsQIxKlVh9hKp5vLW5scY7iIgRFO1bfAUNRQXyr0nAECk-ulVB39uW4E3J~G8hAT4e93alCnhrecLQgNeIGjdfcvqQRqwo1gTbqtofQK3D0qJBmqKTgE8KbSUME1x7tnlfc1~m0CZqBlm8uOgMI5AwGkXdMfQl3weYYwnenDikLqrDm5XrYOjUosnoPFzAqoeIntvfmCyIKE3db5UQo8IwiWrzyXfBDJnlUhy6NkHKt6hSAh3mhR4BATQgHeKICf0uqaonDyAc7aaSaWuupa9ngeSr9njgj9xj55Gzg5tA__"
#tar -xvf xeniumranger-3.1.0.tar.gz
#rm xeniumranger-3.1.0.tar.gz
#export PATH=~/files/xeniumranger-xenium3.1:$PATH

#Patch output-XETG00217__0038291__Region_2__20241212__142808
python patch_transcript_assignments.py \
--xenium-bundle ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_2__20241212__142808/ \
--viz-polygons ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_2__20241212__142808/Proseg/baysor-cell-polygons.geojson \
--transcript-assignment ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_1__20241212__142808/Proseg/baysor-transcript-metadata.csv \
--output-transcript-assignment ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_1__20241212__142808/Proseg/new-baysor-transcript-metadata.csv \
--output-viz-polygons ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_2__20241212__142808/Proseg/new-baysor-cell-polygons.geojson 

python patch_transcript_assignments.py \
--xenium-bundle .. \
--viz-polygons baysor-cell-polygons.geojson \
--transcript-assignment baysor-transcript-metadata.csv \
--output-transcript-assignment new-baysor-transcript-metadata.csv \
--output-viz-polygons new-baysor-cell-polygons.geojson 


./xeniumranger-xenium3.1/xeniumranger import-segmentation --id resegmented \
--xenium-bundle ./Xenium/20241206_KristinaRawData/output-XETG00217__0038213__Region_1__20241206__182124/ \
--viz-polygons ./Xenium/20241206_KristinaRawData/output-XETG00217__0038213__Region_1__20241206__182124/Proseg/baysor-cell-polygons.geojson \
--transcript-assignment ./Xenium/20241206_KristinaRawData/output-XETG00217__0038213__Region_1__20241206__182124/Proseg/baysor-transcript-metadata.csv \
--units microns

./xeniumranger-xenium3.1/xeniumranger import-segmentation --id resegmented11 \
--xenium-bundle ./Xenium/20241206_KristinaRawData/output-XETG00217__0038216__Region_1__20241206__182124/ \
--viz-polygons ./Xenium/20241206_KristinaRawData/output-XETG00217__0038216__Region_1__20241206__182124/Proseg/baysor-cell-polygons.geojson \
--transcript-assignment ./Xenium/20241206_KristinaRawData/output-XETG00217__0038216__Region_1__20241206__182124/Proseg/baysor-transcript-metadata.csv \
--units microns

./xeniumranger-xenium3.1/xeniumranger import-segmentation --id resegmented2 \
--xenium-bundle ./Xenium/20241212_KristinaRawData/output-XETG00217__0038290__Region_1__20241212__142808/ \
--viz-polygons ./Xenium/20241212_KristinaRawData/output-XETG00217__0038290__Region_1__20241212__142808/Proseg/baysor-cell-polygons.geojson \
--transcript-assignment ./Xenium/20241212_KristinaRawData/output-XETG00217__0038290__Region_1__20241212__142808/Proseg/baysor-transcript-metadata.csv \
--units microns

./xeniumranger-xenium3.1/xeniumranger import-segmentation --id resegmented3 \
--xenium-bundle ./Xenium/20241212_KristinaRawData/output-XETG00217__0038290__Region_2__20241212__142808/ \
--viz-polygons ./Xenium/20241212_KristinaRawData/output-XETG00217__0038290__Region_2__20241212__142808/Proseg/baysor-cell-polygons.geojson \
--transcript-assignment ./Xenium/20241212_KristinaRawData/output-XETG00217__0038290__Region_2__20241212__142808/Proseg/baysor-transcript-metadata.csv \
--units microns

./xeniumranger-xenium3.1/xeniumranger import-segmentation --id resegmented4 \
--xenium-bundle ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_1__20241212__142808/ \
--viz-polygons ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_1__20241212__142808/Proseg/baysor-cell-polygons.geojson \
--transcript-assignment ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_1__20241212__142808/Proseg/baysor-transcript-metadata.csv \
--units microns

./xeniumranger-xenium3.1/xeniumranger import-segmentation --id resegmented5 \
--xenium-bundle ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_2__20241212__142808/ \
--viz-polygons ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_2__20241212__142808/Proseg/baysor-cell-polygons.geojson \
--transcript-assignment ./Xenium/20241212_KristinaRawData/output-XETG00217__0038291__Region_2__20241212__142808/Proseg/baysor-transcript-metadata.csv \
--units microns


cp ../../../20241212_KristinaRawData/output-XETG00217__0038291__Region_1__20241212__142808/morphology_focus/morphology_focus_0003.ome.tif /home/kli1/files/resegmented4/XENIUM_RANGER_CS/XR_SETUP_INPUTS/fork0/join-u000073341d/files/xr_min_bundle/morphology_focus/


sleep 1000
bsub -e error.txt -o output.txt -n 64 -M 400000 -R "rusage [mem=400000]" ./xeniumranger-xenium3.1/xeniumranger import-segmentation --id resegmented1 --xenium-bundle ./Xenium/20241206_KristinaRawData/output-XETG00217__0038216__Region_1__20241206__182124/ --viz-polygons ./Xenium/20241206_KristinaRawData/output-XETG00217__0038216__Region_1__20241206__182124/Proseg/baysor-cell-polygons.geojson --transcript-assignment ./Xenium/20241206_KristinaRawData/output-XETG00217__0038216__Region_1__20241206__182124/Proseg/baysor-transcript-metadata.csv --units microns


