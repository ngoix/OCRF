#!/bin/bash
for k in $( seq 3 40 ); do
  java -jar elki-0.7.2-SNAPSHOT.jar KDDCLIApplication \
    -dbc.in mouse.csv \
    -algorithm clustering.kmeans.KMedoidsEM \
    -kmeans.k $k \
    -resulthandler ResultWriter -out.gzip \
    -out output/k-$k 
done
