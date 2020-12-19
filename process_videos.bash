#!/bin/bash

for foldername in ./*/*/; do
    cd $foldername;
    ls *.png;
    convert -delay 5 *.png video.mp4;
    rm *.png;
    mv video.mp4 ../"${PWD##*/}".mp4;
    cd ../../;
done

for foldername in ./*/config*/; do
    rm -r $foldername;
done
