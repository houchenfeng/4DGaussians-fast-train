workdir=$1
# echo "[1/8] 提取首帧图像到 ./colmap_tmp/images (输入目录: $workdir)"
# python scripts/extractimages.py "$workdir"
# echo "完成: 提取首帧"
# echo "[2/8] COLMAP 特征提取"
# colmap feature_extractor --database_path ./colmap_tmp/database.db --image_path ./colmap_tmp/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
# echo "完成: 特征提取"
# echo "[3/8] COLMAP 全匹配"
# colmap exhaustive_matcher --database_path ./colmap_tmp/database.db
# echo "完成: 全匹配"
# echo "[4/8] COLMAP 稀疏重建"
# mkdir ./colmap_tmp/sparse
# colmap mapper --database_path ./colmap_tmp/database.db --image_path ./colmap_tmp/images --output_path ./colmap_tmp/sparse
# echo "完成: 稀疏重建"
# echo "[5/8] 拷贝稀疏结果到 $workdir/sparse_"
# mkdir -p "$workdir/sparse_"
# cp -r ./colmap_tmp/sparse/0/* "$workdir/sparse_"
# echo "完成: 拷贝稀疏结果"

# echo "[6/8] COLMAP 稠密重建"
# mkdir ./colmap_tmp/dense
# colmap image_undistorter --image_path ./colmap_tmp/images --input_path ./colmap_tmp/sparse/0 --output_path ./colmap_tmp/dense --output_type COLMAP
# colmap patch_match_stereo --workspace_path ./colmap_tmp/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
# colmap stereo_fusion --workspace_path ./colmap_tmp/dense --workspace_format COLMAP --input_type geometric --output_path ./colmap_tmp/dense/fused.ply
# echo "完成: 稠密重建 (fused.ply)"

echo "[7/8] 下采样点云 -> $workdir/points3D_multipleview.ply"
python scripts/downsample_point.py ./colmap_tmp/dense/fused.ply "./points3D_multipleview.ply"
echo "完成: 点云下采样"

echo "[8/8] 计算相机位姿 (LLFF)"
git clone https://github.com/Fyusion/LLFF.git
pip install scikit-image
python LLFF/imgs2poses.py ./colmap_tmp/

# cp ./colmap_tmp/poses_bounds.npy "$workdir/poses_bounds_multipleview.npy"
# echo "完成: 位姿文件写入 $workdir/poses_bounds_multipleview.npy"

echo "清理临时目录"
# rm -rf ./colmap_tmp
# rm -rf ./LLFF
echo "全部完成"



