#! /bin/bash

indir=images
outdir=output
rm -rf ${outdir}
mkdir ${outdir}
confidence_threshold=0.15

for image in $(ls ./${indir}/); do
	echo "Detecting objects on ${image}"
	python3 inference.py -c1 ./vanilla.pb -c2 ./multires.pb --input ./${indir}/${image} --output ./${outdir}/${image}.txt > /dev/null 2> /dev/null
	python3 show_regions.py ./${indir}/${image} ./${outdir}/${image}.txt ./${outdir}/${image} ${confidence_threshold}
	echo "DONE"
	display ./${outdir}/${image}
done
