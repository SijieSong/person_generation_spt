CUDA_VISIBLE_DEVICES=4 python2 test_demo.py --dataroot ./imgs --annotation_file './imgs/keypoints.csv'  --name demo_model --model PoseGuidedGenerationDemo --phase test --no_dropout --results_dir './results/demo_test' --dataset_mode 'pose_globallocal_face'  --how_many 6 --which_epoch 'latest'


