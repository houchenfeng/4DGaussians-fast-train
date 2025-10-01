# conda activate 4dgs


# bash multipleviewprogress.sh /home/lt/2024/data/N3D/multipleview/sear_steak

python train.py -s /home/lt/2024/data/N3D/multipleview/sear_steak --port 6017 \
 --expname "dynerf/sear_steak-test3" --configs arguments/dynerf/sear_steak.py 

python render.py --model_path "output/dynerf/sear_steak-test/"  --skip_train --configs arguments/dynerf/sear_steak.py
python render.py --model_path "output/dynerf/sear_steak/"  --skip_train --configs arguments/dynerf/sear_steak.py
python render.py --model_path "output/dynerf/sear_steak/"  --skip_test --configs arguments/dynerf/sear_steak.py

python metrics.py --model_path "output/dynerf/sear_steak/" 

./viewers/bin/SIBR_remoteGaussian_app.exe --port 6017 # port should be same with your trainging code.