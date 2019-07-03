# 完整YOLOv3流程

原作者 :  [YunYang1994](https://github.com/YunYang1994)

论文：	[YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

环境：	win7 + anaconda + python3.6 ( 建议用anaconda，方便一些)

下载代码之前，请先装好环境和相关依赖。

```
pip install -r docs/requirements.txt
```



## 1.github下载YOLOv3的tensorflow实现代码。

``` sda 
git clone git@github.com:sowei728/tensorflow_for_YOLOv3.git
```



## 2.下载YOLOv3的初始权值放到checkpoint文件夹中。

```
wget https://pjreddie.com/media/files/yolov3.weights
```

windows 没有wget命令就直接找个浏览器把URL输进去就好了



## 3.标注自己的VOC格式数据。

使用labelImg脚本，下载地址： http://tzutalin.github.io/labelImg/

脚本使用非常简单，自己看看就会了，最后脚本生成的xml文件放入dataset文件夹的anno文件夹中，原jpg的图片放入images文件夹中



## 4. 将xml文件中的标注信息汇总到txt中。

自己新建3个txt文件：

	labels.txt  	存放所有图片的标注信息
	
	train.txt  	存放训练图片的标注信息（总图片的80%）
	
	test.txt  	存放测试图片的标注信息（总图片的20%）

文件内容：(以空格隔开)

	图片路径 x1min y1min x1max y1max cls1_id x2min y2min x2max y2max cls2_id	........

运行脚本即可：

```
python scripts/load_xml_into_txt.py
```



## 5.将train.txt和test.txt转化为train.tfrecords和test.tfrecords文件。

```
python convert_tfrecord.py --dataset_txt ./apple_dataset/train.txt --tfrecord_path_prefix ./apple_dataset/apple_train

python convert_tfrecord.py --dataset_txt ./apple_dataset/test.txt  --tfrecord_path_prefix ./apple_dataset/apple_test
```

apple_test.tfrecords 和 apple_train.tfrecords会生成在dataset目录下



## 6. 查看文件标注有无问题。

```
python show_input_image.py 
```

如果正常出现图像及标注信息则代表文件标注无问题



## 7.kmeans求出k个anchor box的长宽。

```
python kmeans.py  
```

会输出准确率和k个anchor box的长宽信息，本实验中确定的k值为9

长宽信息需要填入到**data/apple_anchors.txt**中



## 8.转化yolov3.weights为ckpt模型权值。

```
python convert_weight.py --convert
```



## 9.训练模型，保存模型到checkpoint文件夹中。

```
python quick_train.py
```



## 10. 将模型格式转化为pb格式。

```
python convert_weight.py -cf ./checkpoint/yolov3.ckpt-2500 -nc 5 -ap ./data/apple_anchors.txt --freeze
```

这里面的2500表示实际模型的迭代次数，要根据实际模型来更改

5表示具体的分类总数

具体分类信息见**data/apple.names**文件



## 11.模型对单张图片进行测试。

```
python quick_test.py
```



## 12.模型对整体数据集的精度和召回率进行评估。

```
python evaluate.py
```

evaluate1.py是我根据我论文重写的一个评估函数，有兴趣可以和我讨论。