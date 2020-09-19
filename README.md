# Tensorflow2-ObjectDetectionAPI-Colab-Hands-On


# Directory
<pre>
│  
├─01_train_data─┬─000000.jpg
│               │     :
│               └─000049.jpg
│      
├─02_tfrecord
│      
├─03_pretrained_model─efficientdet_d0_coco17_tpu-32─┬─pipeline.config
│                                                   ├─checkpoint──┬─checkpoint
│                                                   │             ├─ckpt-0.data-00000-of-00001
│                                                   │             └─ckpt-0.index
│                                                   └─saved_model─┬─saved_model.pb
│                                                                 └─variables─┬─variables.data-00000-of-00001
│                                                                             └─variables.index
│
└─04_test_data─┬─000050.jpg
               │     :
               └─000099.jpg
</pre>

# パイプラインコンフィグ修正箇所
3行目：num_classes: 90 → 1<br>
134行目：batch_size: 128 → 4<br>
161行目：fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED" → "/content/models/research/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/03_pretrained_model/efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0"
167行目：fine_tune_checkpoint_type: "classification" → "detection"<br>
168行目：use_bfloat16: true → false<br>
172行目：label_map_path: "PATH_TO_BE_CONFIGURED/label_map.txt" → "/content/models/research/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/02_tfrecord/tf_label_map.pbtxt"<br>
174行目：input_path: "PATH_TO_BE_CONFIGURED/train2017-?????-of-00256.tfrecord" → "/content/models/research/train_data/??????.tfrecord"<br>
185行目：label_map_path: "PATH_TO_BE_CONFIGURED/label_map.txt" → "/content/models/research/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/02_tfrecord/tf_label_map.pbtxt"<br>
189行目：input_path: "PATH_TO_BE_CONFIGURED/val2017-?????-of-00032.tfrecord" → "/content/models/research/val_data/??????.tfrecord"


# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
FingerFrameDetection-TF2 is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
