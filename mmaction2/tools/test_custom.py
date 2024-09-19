from mmaction.apis import inference_recognizer, init_recognizer

config_path = 'work_dirs/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_sgv/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_sgv.py'
checkpoint_path = 'work_dirs/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_sgv/best_acc_top1_epoch_24.pth' # can be a local path
img_path = 'data_train_base/Suturing/case_001_video_part_001_Suturing_1_12_49_to_1_30_20.mp4'   # you can specify your own picture path

# build the model from a config file and a checkpoint file
model = init_recognizer(config_path, checkpoint_path, device="cpu")  # device can be 'cuda:0'
# test a single image
result = inference_recognizer(model, img_path)

print(result)