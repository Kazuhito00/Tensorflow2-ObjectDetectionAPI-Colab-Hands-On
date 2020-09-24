# Tensorflow2-ObjectDetectionAPI-Colab-Hands-On
Tensorflow2 Object Detection APIのハンズオン用資料です。<br>
VoTTでのアノテーションをローカルPCで実施し、学習～推論はColaboratory上で実施します。<br><br>
以下の内容を含みます。<br>
* 学習用データセット ※アノテーション未実施
* テスト用データセット
* ファインチューニング用モデル(EffientDet D0)
* Colaboratory用スクリプト(環境設定、モデル訓練、推論結果確認)

<details>
<summary>Directory</summary>

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
Colaboratory用スクリプト(環境設定、モデル訓練、推論結果確認)

#### 01_train_data
学習用データセット ※アノテーション未実施

#### 02_tfrecord
アノテーション実施済みTFRecord格納先

#### 03_pretrained_mode
ファインチューニング用モデル(EffientDet D0)

#### 04_test_data
テスト用データセット

</details>


# Overview
2時間程度のボリュームの想定です。
1. VoTT：アノテーション(約30～60分)
1. Colaboratory：Object Detection API設定
1. パイプラインコンフィグ修正
1. Colaboratory：モデル訓練(約25分)
1. Colaboratory：推論

# Preparations
事前準備として以下が必要です。
* [VoTT](https://github.com/microsoft/VoTT)のインストール
* Googleアカウント(Google Colaboratory、Googleドライブで使用)

# 1. VoTT：アノテーション

<details>
<summary>VoTT</summary>

![2020-09-19 (3)](https://user-images.githubusercontent.com/37477845/94047557-38407600-fe0d-11ea-8d10-041a27546e85.png)
![2020-09-19 (4)](https://user-images.githubusercontent.com/37477845/94047561-3971a300-fe0d-11ea-8bd2-4bd621cd531c.png)
![2020-09-19 (6)](https://user-images.githubusercontent.com/37477845/94047562-3a0a3980-fe0d-11ea-8619-7dab9d63160b.png)
![2020-09-19 (7)](https://user-images.githubusercontent.com/37477845/94047564-3aa2d000-fe0d-11ea-9aea-b66aab732841.png)
![2020-09-19 (8)](https://user-images.githubusercontent.com/37477845/94047566-3b3b6680-fe0d-11ea-8534-8402652d9f32.png)
![2020-09-19 (9)](https://user-images.githubusercontent.com/37477845/94047569-3bd3fd00-fe0d-11ea-958d-745d86d3436f.png)
![2020-09-19 (10)](https://user-images.githubusercontent.com/37477845/94047571-3c6c9380-fe0d-11ea-94fb-94a4a4dd4467.png)
![2020-09-19 (11)](https://user-images.githubusercontent.com/37477845/94047572-3d052a00-fe0d-11ea-80cb-e6b2f39fbfc9.png)
![2020-09-19 (12)](https://user-images.githubusercontent.com/37477845/94047577-3d9dc080-fe0d-11ea-9f4f-b5fe7727fc12.png)
![2020-09-19 (13)](https://user-images.githubusercontent.com/37477845/94047578-3e365700-fe0d-11ea-86b9-2d88ef24d0c0.png)
![2020-09-19 (14)](https://user-images.githubusercontent.com/37477845/94047588-41314780-fe0d-11ea-9574-0cb6c77f8be5.png)
![2020-09-19 (15)](https://user-images.githubusercontent.com/37477845/94047598-442c3800-fe0d-11ea-9285-d72713520a65.png)
![2020-09-19 (16)](https://user-images.githubusercontent.com/37477845/94047601-44c4ce80-fe0d-11ea-89fc-92b86e4ba3b8.png)
![2020-09-19 (17)](https://user-images.githubusercontent.com/37477845/94047603-44c4ce80-fe0d-11ea-8c0d-3ebc2e740560.png)


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

# 2. Colaboratory：Object Detection API設定

# 3. パイプラインコンフィグ修正

# 4. Colaboratory：モデル訓練

# 5. Colaboratory：推論

<!--
# パイプラインコンフィグ修正箇所
3行目：num_classes: 90 → 1<br>
134行目：batch_size: 128 → 16<br>
161行目：fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED" → "/content/models/research/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/03_pretrained_model/efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0"
167行目：fine_tune_checkpoint_type: "classification" → "detection"<br>
168行目：use_bfloat16: true → false<br>
172行目：label_map_path: "PATH_TO_BE_CONFIGURED/label_map.txt" → "/content/models/research/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/02_tfrecord/tf_label_map.pbtxt"<br>
174行目：input_path: "PATH_TO_BE_CONFIGURED/train2017-?????-of-00256.tfrecord" → "/content/models/research/train_data/??????.tfrecord"<br>
185行目：label_map_path: "PATH_TO_BE_CONFIGURED/label_map.txt" → "/content/models/research/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/02_tfrecord/tf_label_map.pbtxt"<br>
189行目：input_path: "PATH_TO_BE_CONFIGURED/val2017-?????-of-00032.tfrecord" → "/content/models/research/val_data/??????.tfrecord"
-->

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
FingerFrameDetection-TF2 is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
