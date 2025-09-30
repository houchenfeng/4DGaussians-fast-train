# conda activate 4dgs


# bash multipleviewprogress.sh /home/lt/2024/data/N3D/multipleview/sear_steak

python train.py -s /home/lt/2024/data/N3D/multipleview/sear_steak --port 6017 \
 --expname "dynerf/sear_steak" --configs arguments/dynerf/sear_steak.py 