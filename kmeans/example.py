#encoding:utf-8
import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import Kmeans, avg_iou

ANNOTATIONS_PATH = "../apple_dataset/anno"
CLUSTERS = 9

def load_dataset(path):
	dataset = []
	res = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			xmin = int(obj.findtext("bndbox/xmin")) / width
			ymin = int(obj.findtext("bndbox/ymin")) / height
			xmax = int(obj.findtext("bndbox/xmax")) / width
			ymax = int(obj.findtext("bndbox/ymax")) / height

			dataset.append([xmax - xmin, ymax - ymin])


	return np.array(dataset)

# 所有目标框的[相对长，相对宽]
data= load_dataset(ANNOTATIONS_PATH)
print("data.shape:",data.shape[0])
out = Kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
out_list = []
for i in range(len(out)):
	out_list.append(out[i][0])
	out_list.append(out[i][1])

print("Boxes:\n {}".format(np.array(out_list)))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))