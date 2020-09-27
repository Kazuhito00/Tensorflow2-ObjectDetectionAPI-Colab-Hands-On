[Japanese/[English](https://github.com/Kazuhito00/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/blob/master/README_EN.md)] 
# Tensorflow2-ObjectDetectionAPI-Colab-Hands-On
![mkv4t-6ilnu](https://user-images.githubusercontent.com/37477845/94301550-b46dc180-ffa5-11ea-8a1c-7fdf14278cd9.gif)

Tensorflow2 Object Detection APIのハンズオン用資料です。<br>
VoTTでのアノテーションをローカルPCで実施し、学習～推論はColaboratory上で実施します。<br><br>
以下の内容を含みます。<br>
* 学習用データセット ※アノテーション未実施
* テスト用データセット
* ファインチューニング用モデル(EffientDet D0)
* Colaboratory用スクリプト(環境設定、モデル訓練、推論結果確認)

<details>
<summary>ディレクトリ構成</summary>

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

# Requirement
Tensorflow 2.3.0

# Overview
2時間程度のボリュームの想定です。
1. VoTT：アノテーション(約30～60分)
1. Colaboratory：Object Detection API設定
1. パイプラインコンフィグ修正
1. Colaboratory：モデル訓練(約25分)
1. Colaboratory：推論

# Preparations
事前準備として以下が必要です。
* このリポジトリのローカル環境へのクローン
* [VoTT](https://github.com/microsoft/VoTT)のインストール
* Googleアカウント(Google Colaboratory、Googleドライブで使用)

# 1. VoTT：アノテーション
[VoTT](https://github.com/microsoft/VoTT)を使用してアノテーションを行い、TFRecord形式で出力します。

<details>
<summary>VoTTのプロジェクト設定</summary>
	
#### 「新規プロジェクト」を選択する
![2020-09-19 (3)](https://user-images.githubusercontent.com/37477845/94047557-38407600-fe0d-11ea-8d10-041a27546e85.png)
#### プロジェクト設定を行う
表示名：Tensorflow2-ObjectDetectionAPI-Colab-Hands-On<br>
セキュリティトークン：Generate New Security Token<br>
ソース接続：「Add Connection」を押下<br>
![2020-09-19 (4)](https://user-images.githubusercontent.com/37477845/94047561-3971a300-fe0d-11ea-8bd2-4bd621cd531c.png)
#### ソース接続の接続設定を行う
表示名：Tensorflow2-ObjectDetectionAPI-Colab-Hands-On-TrainData
![2020-09-19 (6)](https://user-images.githubusercontent.com/37477845/94047562-3a0a3980-fe0d-11ea-8619-7dab9d63160b.png)
プロバイダー：ローカルファイルシステム
![2020-09-19 (7)](https://user-images.githubusercontent.com/37477845/94047564-3aa2d000-fe0d-11ea-9aea-b66aab732841.png)
フォルダーパス：クローンしたリポジトリの「01_train_data」ディレクトリを指定
![2020-09-19 (8)](https://user-images.githubusercontent.com/37477845/94047566-3b3b6680-fe0d-11ea-8534-8402652d9f32.png)
#### ターゲット接続の接続設定を行う
ターゲット接続：Add Connection
![2020-09-19 (9)](https://user-images.githubusercontent.com/37477845/94047569-3bd3fd00-fe0d-11ea-958d-745d86d3436f.png)
表示名：Tensorflow2-ObjectDetectionAPI-Colab-Hands-On-TFRecord<br>
プロバイダー：ローカルファイルシステム<br>
フォルダーパス：クローンしたリポジトリの「02_tfrecord」ディレクトリを指定<br>
![2020-09-19 (10)](https://user-images.githubusercontent.com/37477845/94047571-3c6c9380-fe0d-11ea-94fb-94a4a4dd4467.png)
<!-- #### 8
![2020-09-19 (11)](https://user-images.githubusercontent.com/37477845/94047572-3d052a00-fe0d-11ea-80cb-e6b2f39fbfc9.png)-->
#### タグを追加し設定を保存する
タグ：「Fish」を追加<br>
「プロジェクトを保存」を押下
![94047577-3d9dc080-fe0d-11ea-9f4f-b5fe7727fc12](https://user-images.githubusercontent.com/37477845/94283906-98a9f180-ff8c-11ea-9e16-a546b26ba763.png)
</details>

<details>
<summary>VoTTを使用してアノテーションを実施</summary>
	
#### マウス左ドラッグで魚を選択する
![2020-09-19 (13)](https://user-images.githubusercontent.com/37477845/94047578-3e365700-fe0d-11ea-86b9-2d88ef24d0c0.png)
#### TAGSから「Fish」を選択する
南京錠のマークを選択しておくことでタグを使用するタグを固定することが可能
![2020-09-19 (14)](https://user-images.githubusercontent.com/37477845/94047588-41314780-fe0d-11ea-9574-0cb6c77f8be5.png)
<!-- #### 12
![2020-09-19 (15)](https://user-images.githubusercontent.com/37477845/94047598-442c3800-fe0d-11ea-9285-d72713520a65.png)-->
</details>

<details>
<summary>TFRecordエクスポート</summary>
	
#### エクスポート設定
プロバイダー：Tensorflow レコード<br>
アセットの状態：タグ付きアセットのみ<br>
「エクスポート設定を保存」を押下する
![2020-09-19 (16)](https://user-images.githubusercontent.com/37477845/94047601-44c4ce80-fe0d-11ea-89fc-92b86e4ba3b8.png)
アノテーション画面からエクスポートマークを押下し、TFRecordをエクスポートする。
![2020-09-19 (14)](https://user-images.githubusercontent.com/37477845/94047588-41314780-fe0d-11ea-9574-0cb6c77f8be5.png)
</details>

<details>
<summary>注意事項（詳細確認中）</summary>

画像の端の対象をアノテーションする際に、以下のように端から少し隙間を設けてください。
![2020-09-19 (17)](https://user-images.githubusercontent.com/37477845/94047603-44c4ce80-fe0d-11ea-8c0d-3ebc2e740560.png)<br>
問題の詳細は確認中ですが、隙間を開けずにアノテーションをすると、<br>
VoTTの問題かTensorflowの問題か、モデル学習時に以下のエラーが発生します。
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/blob/master/[Colaboratory]Tensorflow2-ObjectDetectionAPI-Colab-Hands-On.ipynb)<br>
以降の作業はGoogle Colaboratory上で実施します。※パイプラインコンフィグ修正をのぞく<br>
[Open In Colab]リンクからノートブックを開き、以下の順に実行してください。
* Google Driveマウント
* Tensorflow Object Detection API設定
* Tensorflow2-ObjectDetectionAPI-Colab-Hands-Onリポジトリクローン

# 3.TFRecordアップロード
「Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/02_tfrecord」に<br>VoTTからエクスポートしたTFRecordとtf_label_map.pbtxtを格納してください。<br>
格納後、以下を実行してください。
* 学習データ/検証データ 分割

# 4. パイプラインコンフィグ修正
「03_pretrained_model\efficientdet_d0_coco17_tpu-32\pipeline.config」のパイプラインコンフィグを以下のように修正して、<br>
Colaboratory上の「Tensorflow2-ObjectDetectionAPI-Colab-Hands-On/03_pretrained_model」にアップロードしてください。<br>
<details>
<summary>パイプラインコンフィグ修正箇所</summary>

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
<summary>パイプラインコンフィグ修正箇所 ※余裕のある方向け</summary>

パイプラインコンフィグにはデータ拡張設定も記載されています。<br>
初期のパイプラインコンフィグには、以下の水平反転、ランダムスケールクロップのみのデータ拡張が設定されています。<br>
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

使用可能なデータ拡張手法は、[preprocessor.proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto)、[preprocessor.py](https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py)に記載されているため、<br>
必要に応じて追加してみてください。
</details>

# 5. Colaboratory：モデル訓練
以下の順に実行してください。
* Googleドライブに保存先ディレクトリを作成
* TensorBoard
* 学習
* saved model形式へエクスポート

# 6. Colaboratory：推論
以下の順に実行してください。
* モデルロード
* 推論
* 推論結果確認

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Tensorflow2-ObjectDetectionAPI-Colab-Hands-On is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
