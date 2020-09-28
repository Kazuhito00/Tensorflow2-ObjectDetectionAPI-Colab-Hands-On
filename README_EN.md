[[Japanese](https://github.com/Kazuhito00/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On)/English] 
# Tensorflow2-ObjectDetectionAPI-Colab-Hands-On
![mkv4t-6ilnu](https://user-images.githubusercontent.com/37477845/94301550-b46dc180-ffa5-11ea-8a1c-7fdf14278cd9.gif)

Hands-on documentation for the Tensorflow2 Object Detection API.<br>
Annotation with VoTT is performed on the local PC, and learning-inference is performed on Colaboratory.<br><br>
This repository contains the following:<br>
* Dataset for learning (Annotation not implemented)
* Test dataset
* Model for fine-tuning(EffientDet D0)
* Script for Google Colaboratory(Environment setting, model training, inference result confirmation)

<details>
<summary>Directory structure</summary>

<pre>
│ [Colaboratory]Tensorflow2_ObjectDetectionAPI_Colab_Hands_On.ipynb
|
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

#### [Colaboratory]Tensorflow2_ObjectDetectionAPI_Colab_Hands_On.ipynb
Script for Google Colaboratory(Environment setting, model training, inference result confirmation)

#### 01_train_data
Dataset for learning (Annotation not implemented)

#### 02_tfrecord
Annotated TFRecord storage location

#### 03_pretrained_mode
Model for fine-tuning(EffientDet D0)

#### 04_test_data
Test dataset

</details>

# Requirement
Tensorflow 2.3.0

# Overview
This hands-on assumes about 2 hours.
1. VoTT：Annotation(30-60minutes)
1. Colaboratory：Object Detection API Setting
1. Pipeline-config correction
1. Colaboratory：Model training(About 25minutes)
1. Colaboratory：Inference

# Preparations
The following is required as a preliminary preparation.
* Clone this repository to your local PC.
* [VoTT](https://github.com/microsoft/VoTT) installation.
* Google account(Used in Google Drive, Google Colaboratory)

# 1. VoTT：Annotation
Annotate using [VoTT](https://github.com/microsoft/VoTT) and output in TFRecord format.

<details>
<summary>VoTT project settings</summary>
	
#### Select "New Project"
![2020-09-19 (3)](https://user-images.githubusercontent.com/37477845/94047557-38407600-fe0d-11ea-8d10-041a27546e85.png)
#### Make project settings
Display name：Tensorflow2-ObjectDetectionAPI-Colab-Hands-On<br>
Security token：Generate New Security Token<br>
Source connection：「Add Connection」を押下<br>
![2020-09-19 (4)](https://user-images.githubusercontent.com/37477845/94047561-3971a300-fe0d-11ea-8bd2-4bd621cd531c.png)
#### Set the connection of the source connection
Display name：Tensorflow2-ObjectDetectionAPI-Colab-Hands-On-TrainData
![2020-09-19 (6)](https://user-images.githubusercontent.com/37477845/94047562-3a0a3980-fe0d-11ea-8619-7dab9d63160b.png)
Provider: Local file system
![2020-09-19 (7)](https://user-images.githubusercontent.com/37477845/94047564-3aa2d000-fe0d-11ea-9aea-b66aab732841.png)
Folder path：Specify the "01_train_data" directory of the cloned repository
![2020-09-19 (8)](https://user-images.githubusercontent.com/37477845/94047566-3b3b6680-fe0d-11ea-8534-8402652d9f32.png)
#### Set the connection of the target connection
Target connection：Add Connection
![2020-09-19 (9)](https://user-images.githubusercontent.com/37477845/94047569-3bd3fd00-fe0d-11ea-958d-745d86d3436f.png)
Display name：Tensorflow2-ObjectDetectionAPI-Colab-Hands-On-TFRecord<br>
Provider: Local file system<br>
Folder path：Specify the "02_tfrecord" directory of the cloned repository<br>
![2020-09-19 (10)](https://user-images.githubusercontent.com/37477845/94047571-3c6c9380-fe0d-11ea-94fb-94a4a4dd4467.png)
<!-- #### 8
![2020-09-19 (11)](https://user-images.githubusercontent.com/37477845/94047572-3d052a00-fe0d-11ea-80cb-e6b2f39fbfc9.png)-->
#### Add tags and save settings
Tags：Add "Fish"<br>
Press "Save Project"
![94047577-3d9dc080-fe0d-11ea-9f4f-b5fe7727fc12](https://user-images.githubusercontent.com/37477845/94283906-98a9f180-ff8c-11ea-9e16-a546b26ba763.png)
</details>

<details>
<summary>Annotate using VoTT</summary>
	
#### Select a fish by left dragging the mouse
![2020-09-19 (13)](https://user-images.githubusercontent.com/37477845/94047578-3e365700-fe0d-11ea-86b9-2d88ef24d0c0.png)
#### Select "Fish" from TAGS
You can lock the tag you want to use by selecting the padlock mark.
![2020-09-19 (14)](https://user-images.githubusercontent.com/37477845/94047588-41314780-fe0d-11ea-9574-0cb6c77f8be5.png)
<!-- #### 12
![2020-09-19 (15)](https://user-images.githubusercontent.com/37477845/94047598-442c3800-fe0d-11ea-9285-d72713520a65.png)-->
</details>

<details>
<summary>TFRecord export</summary>
	
#### Export settings
Provider：Tensorflow record<br>
Asset status: Tagged assets only<br>
Click "Save Export Settings"
![2020-09-19 (16)](https://user-images.githubusercontent.com/37477845/94047601-44c4ce80-fe0d-11ea-89fc-92b86e4ba3b8.png)
Click the export icon from the annotation screen to export TFRecord.
![2020-09-19 (14)](https://user-images.githubusercontent.com/37477845/94047588-41314780-fe0d-11ea-9574-0cb6c77f8be5.png)
</details>

<details>
<summary>Precautions (details are being confirmed)</summary>

When annotating the target at the edge of the image, leave a small gap from the edge as shown below.
![2020-09-19 (17)](https://user-images.githubusercontent.com/37477845/94047603-44c4ce80-fe0d-11ea-8c0d-3ebc2e740560.png)<br>
I am checking the details of the problem, but if I annotate without opening a gap, <br>
I do not know whether it is a VoTT problem or a Tensorflow problem, but the following error occurs when training the model.
<pre>
W0921 13:29:32.965700 140050120722176 optimizer_v2.py:1275] Gradients do not exist for variables ['top_bn/gamma:0', 'top_bn/beta:0'] when minimizing the loss.
Traceback (most recent call last):
  File "object_detection/model_main_tf2.py", line 113, in <module>
    tf.compat.v1.app.run()
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/platform/app.py", line 40, in run
    _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
  File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 300, in run
    _run_main(main, args)
  File "/usr/local/lib/python3.6/dist-packages/absl/app.py", line 251, in _run_main
    sys.exit(main(argv))
  File "object_detection/model_main_tf2.py", line 110, in main
    record_summaries=FLAGS.record_summaries)
  File "/usr/local/lib/python3.6/dist-packages/object_detection/model_lib_v2.py", line 639, in train_loop
    loss = _dist_train_step(train_input_iter)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py", line 780, in __call__
    result = self._call(*args, **kwds)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py", line 807, in _call
    return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py", line 2829, in __call__
    return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py", line 1848, in _filtered_call
    cancellation_manager=cancellation_manager)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py", line 1924, in _call_flat
    ctx, args, cancellation_manager=cancellation_manager))
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py", line 550, in call
    ctx=ctx)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/execute.py", line 60, in quick_execute
    inputs, attrs, num_outputs)
tensorflow.python.framework.errors_impl.InvalidArgumentError: 2 root error(s) found.
  (0) Invalid argument:  assertion failed: [[0.15956609][0.103383526][0.109880842]...] [[0.23180081][0.133959055][0.132812485]...]
	 [[{{node Assert_1/AssertGuard/else/_35/Assert_1/AssertGuard/Assert}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]]
	 [[IteratorGetNext]]
	 [[Loss/localization_loss_1/write_summary/summary_cond/pivot_t/_4/_111]]
  (1) Invalid argument:  assertion failed: [[0.15956609][0.103383526][0.109880842]...] [[0.23180081][0.133959055][0.132812485]...]
	 [[{{node Assert_1/AssertGuard/else/_35/Assert_1/AssertGuard/Assert}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]]
	 [[IteratorGetNext]]
</pre>
</details>

# 2. Colaboratory：Object Detection API Setting
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/blob/master/[Colaboratory]Tensorflow2-ObjectDetectionAPI-Colab-Hands-On.ipynb)<br>
Subsequent work will be performed on Google Colaboratory. ※Except for pipeline config modification<br>
Open your notebook from the [Open In Colab] link and run it in the following order.
* Google Drive mount
* Set Tensorflow Object Detection API
* Clone Tensorflow2-ObjectDetectionAPI-Colab-Hands-On repository

# 3.Upload TFRecord
Store the TFRecord exported from VoTT and tf_label_map.pbtxt in "Tensorflow2-Object Detection API-Colab-Hands-On / 02_tfrecord".<br>
After storing, execute the following.
* Split Training data/validation data

# 4. Pipeline-config correction
Modify the pipeline config of "03_pretrained_model/coefficientdet_d0_coco17_tpu-32/pipeline.config" as follows, <br>
Please upload to "Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/03_pretrained_model" on Colaboratory.<br>
<details>
<summary>Pipeline-config correction part</summary>

* 3行目(Line 3)：クラス数(num_classes)<br>変更前(Before) : 90<br>変更後(After) : 1<br>
* 134行目(Line 134)：バッチサイズ(batch_size)<br>変更前(Before) : 128<br>変更後(After) : 16<br>
* 161行目(Line 161)：ファインチューニング用のチェックポイント格納先(fine_tune_checkpoint)<br>変更前(Before) : "PATH_TO_BE_CONFIGURED"<br>変更後(After) : "/content/models/research/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/03_pretrained_model/efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0"
* 167行目(Line 167)：ファインチューニング方法(fine_tune_checkpoint_type)<br>変更前(Before) : "classification"<br>変更後(After) : "detection"<br>
* 168行目(Line 168)：Googleカスタム 16ビットbrain浮動小数点の使用有無(use_bfloat16)<br>変更前(Before) : true<br>変更後(After) : false<br>
* 172行目(Line 172)：ラベルマップファイルの格納先(label_map_path)<br>変更前(Before) : "PATH_TO_BE_CONFIGURED/label_map.txt"<br>変更後(After) : "/content/models/research/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/02_tfrecord/tf_label_map.pbtxt"<br>
* 174行目(Line 174)：学習データの格納先(input_path)<br>変更前(Before) : "PATH_TO_BE_CONFIGURED/train2017-?????-of-00256.tfrecord"<br>変更後(After) : "/content/models/research/train_data/??????.tfrecord"<br>
* 185行目(Line 185)：ラベルマップファイルの格納先(label_map_path)<br>変更前(Before) : "PATH_TO_BE_CONFIGURED/label_map.txt"<br>変更後(After) : "/content/models/research/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/02_tfrecord/tf_label_map.pbtxt"<br>
* 189行目(Line 189)：バリデーションデータの格納先(input_path)<br>変更前(Before) : "PATH_TO_BE_CONFIGURED/val2017-?????-of-00032.tfrecord"<br>変更後(After) : "/content/models/research/val_data/??????.tfrecord"
</details>

<details>
<summary>Pipeline-config correction parts ※Those who can afford</summary>

Data Augmentation settings are also listed in the pipeline config.<br>
In the initial pipeline config, the following horizontal inversion and random scale crop only data augmentation are set.<br>
<pre>
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_scale_crop_and_pad_to_square {
      output_size: 512
      scale_min: 0.10000000149011612
      scale_max: 2.0
    }
  }
</pre>

Available data augmentation techniques are [preprocessor.proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto)、[preprocessor.py](https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py)so <br>
Try adding as needed.
</details>

# 5. Colaboratory：Model training
Please execute in the following order.
* Create directory in Google Drive
* TensorBoard
* Training
* Export to saved-model format

# 6. Colaboratory：Inference
Please execute in the following order.
* Load model
* Inference
* Inference result confirmation

# Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
 
# License 
Tensorflow2-ObjectDetectionAPI-Colab-Hands-On is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
