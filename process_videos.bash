#!/bin/bash

for foldername in ./*/*/; do
    cd $foldername;
    ls *.png;
    convert -delay 5 *.png -loop 0 video.gif;
    rm *.png;
    mv video.gif ../"${PWD##*/}".gif;
    cd ../../;
done

for foldername in ./*/config*/; do
    rm -r $foldername;
done
