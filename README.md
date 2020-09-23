# Tensorflow2-ObjectDetectionAPI-Colab-Hands-On

![2020-09-19 (3)](https://user-images.githubusercontent.com/37477845/94046873-3629e780-fe0c-11ea-9e0f-d060bad6f4fc.png)
![2020-09-19 (4)](https://user-images.githubusercontent.com/37477845/94046880-388c4180-fe0c-11ea-8d12-4cf2ba077016.png)
![2020-09-19 (6)](https://user-images.githubusercontent.com/37477845/94046885-39bd6e80-fe0c-11ea-87cc-f0f7918aa62f.png)
![2020-09-19 (7)](https://user-images.githubusercontent.com/37477845/94046887-3a560500-fe0c-11ea-8aff-d5125609da72.png)
![2020-09-19 (8)](https://user-images.githubusercontent.com/37477845/94046890-3b873200-fe0c-11ea-9729-7fd81a4d20b4.png)
![2020-09-19 (9)](https://user-images.githubusercontent.com/37477845/94046892-3cb85f00-fe0c-11ea-984e-009c68560b70.png)
![2020-09-19 (10)](https://user-images.githubusercontent.com/37477845/94046894-3de98c00-fe0c-11ea-8de8-a08d40deb1c5.png)
![2020-09-19 (11)](https://user-images.githubusercontent.com/37477845/94046897-3f1ab900-fe0c-11ea-9527-18ae7e963a3f.png)
![2020-09-19 (12)](https://user-images.githubusercontent.com/37477845/94046905-404be600-fe0c-11ea-86a5-bb229cd2024e.png)
![2020-09-21 (0)](https://user-images.githubusercontent.com/37477845/94046916-42ae4000-fe0c-11ea-8c08-a9cc3f4df21f.png)
![2020-09-21 (1)](https://user-images.githubusercontent.com/37477845/94046922-46da5d80-fe0c-11ea-9566-e917f2c38eb9.png)
![2020-09-19 (15)](https://user-images.githubusercontent.com/37477845/94046907-40e47c80-fe0c-11ea-8f8e-4a6e9361548b.png)
![2020-09-19 (16)](https://user-images.githubusercontent.com/37477845/94046913-4215a980-fe0c-11ea-8100-9bd71ae96a4b.png)


# Directory
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

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
FingerFrameDetection-TF2 is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
