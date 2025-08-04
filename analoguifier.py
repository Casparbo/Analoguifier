import cv2
import argparse

import numpy as np
from scipy.special import expit


def main():
	args = get_cmd_args()
	img = cv2.imread(args.file)
	print(img.shape)
	smal = cv2.resize(img, tuple(int(s*args.scale) for s in img.shape[:2])[::-1])

	min_thresh, max_thresh, contrast = args.blacken, args.whiten, args.contrast

	if args.gui:
		min_thresh, max_thresh, contrast = create_gui(smal, min_thresh=min_thresh, max_thresh=max_thresh, contrast=contrast)

	result = s_analoguify(img, max_thresh=max_thresh, min_thresh=min_thresh, contrast=contrast/10)
	cv2.imwrite(args.output, result)


def get_cmd_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("file", help="The image file which you wish to analoguify")
	parser.add_argument("--gui", help="Show the gui with sliders to manually adjust each value", action="store_true")
	parser.add_argument("-s", "--scale", help="Scale the image by the factor specified", type=float, default=1)
	parser.add_argument("-b", "--blacken", help="Set the 'blacken' value to this", type=int, choices=range(256), default=0)
	parser.add_argument("-w", "--whiten", help="Set the 'whiten' value to this", type=int, choices=range(256), default=255)
	parser.add_argument("-c", "--contrast", help="Set the 'contrast' value to this", type=int, choices=range(101), default=100)
	parser.add_argument("-o", "--output", help="Path to the output file", default="analog_result.jpg")

	return parser.parse_args()


def create_gui(img, *, min_thresh, max_thresh, contrast):
	cv2.imshow("Test", img)

	cv2.createTrackbar("blacken", "Test", 0, 255, slider(img))
	cv2.createTrackbar("whiten", "Test", 0, 255, slider(img))
	cv2.createTrackbar("contrast", "Test", 1, 100, slider(img))

	cv2.setTrackbarPos("whiten", "Test", max_thresh)
	cv2.setTrackbarPos("blacken", "Test", min_thresh)
	cv2.setTrackbarPos("contrast", "Test", contrast)
	cv2.waitKey(0)

	return get_trackbar_values()


def slider(img):
	def update_img(_):
		min_thresh, max_thresh, contrast = get_trackbar_values()

		cpy = s_analoguify(img, max_thresh=max_thresh, min_thresh=min_thresh, contrast=contrast/10)
		cv2.imshow("Test", cpy)

	return update_img


def get_trackbar_values():
	min_thresh = cv2.getTrackbarPos("blacken", "Test")
	max_thresh = cv2.getTrackbarPos("whiten", "Test")
	contrast = cv2.getTrackbarPos("contrast", "Test")

	return min_thresh, max_thresh, contrast


def s_analoguify(img, *, max_thresh, min_thresh, contrast):
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	l = l.astype(np.float64)

	norm = (l - min_thresh) / max(max_thresh - min_thresh, 1)
	norm = np.clip(norm, 0, 1)
	sig = expit(contrast * (norm - 0.5))
	l = sig * 255
	l = np.clip(l, 0, 255).astype(np.uint8)

	lab = cv2.merge((l, a, b))
	bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

	return bgr


def analoguify(img, *, min_thresh, max_thresh, contrast):
	rad = radicalize(img, min_thresh=min_thresh, max_thresh=max_thresh)
	nrm = normalize(rad, max_thresh, min_thresh)

	return nrm


def radicalize(img, *, min_thresh, max_thresh):
	cpy = img.copy()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	above_range = gray >= max_thresh
	below_range = gray <= min_thresh
	cpy[above_range] = [255, 255, 255]
	cpy[below_range] = [0, 0, 0]

	return cpy


def normalize(img, max_thresh, min_thresh):
	cpy = img.copy().astype(np.float32)

	cpy = (cpy - min_thresh) / max(max_thresh - min_thresh, 1)
	cpy = np.clip(cpy, 0, 1) * 255

	return cpy.astype(np.uint8)


if __name__ == '__main__':
	main()