from matplotlib import pyplot as plt


def imshow(img):
	plt.figure(figsize=(10, 10))
	plt.imshow(img)
	plt.xticks([]); plt.yticks([])