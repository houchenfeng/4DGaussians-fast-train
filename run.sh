# conda activate 4dgs


# bash multipleviewprogress.sh /home/lt/2024/data/N3D/multipleview/sear_steak

CUDA_VISIBLE_DEVICES=0 nohup python train.py -s /home/lt/2024/data/N3D/multipleview/sear_steak --port 6017 --ip 127.0.0.6 \
 --expname "dynerf/sear_steak" --configs arguments/dynerf/sear_steak.py  --debug_mode --quiet > training1.log 2>&1 & 

CUDA_VISIBLE_DEVICES=1 nohup  python train.py -s /home/lt/2024/data/N3D/multipleview/sear_steak --port 6017 --ip 127.0.0.4 \
 --expname "dynerf/sear_steak2" --configs arguments/dynerf/sear_steak.py  --quiet > training2.log 2>&1 & 


nohup python render.py --model_path "output/dynerf/sear_steak/"  --skip_train --configs arguments/dynerf/sear_steak.py > render.log 2>&1 &

nohup python metrics.py --model_path "output/dynerf/sear_steak/" > metrics.log 2>&1 &

# ./viewers/bin/SIBR_remoteGaussian_app.exe --port 6017 # port should be same with your trainging code.