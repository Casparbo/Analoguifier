import cv2
import argparse

import numpy as np

from scipy.special import expit


def main():
	args = get_cmd_args()
	img = cv2.imread(args.file)
	smal = cv2.resize(img, tuple(int(s*args.scale) for s in img.shape[:2])[::-1])

	min_thresh, max_thresh = args.blacken, args.whiten

	if args.gui:
		min_thresh, max_thresh = create_gui(smal, min_thresh=min_thresh, max_thresh=max_thresh)

	result = analoguify(img, max_thresh=max_thresh, min_thresh=min_thresh)
	cv2.imwrite(args.output, result)


def get_cmd_args():
	def check_uint8(value):
		ivalue = int(value)
		if ivalue < 0 or ivalue > 255:
			raise argparse.ArgumentTypeError(f"value should be between 0 and 255, was {value}")
		return ivalue

	parser = argparse.ArgumentParser()
	parser.add_argument("file", help="The image file which you wish to analoguify")
	parser.add_argument("--gui", help="Show the gui with sliders to manually adjust each value", action="store_true")
	parser.add_argument("-s", "--scale", help="Scale the image by the factor specified", type=float, default=1)
	parser.add_argument("-b", "--blacken", help="Set the 'blacken' value to this", type=check_uint8, default=0)
	parser.add_argument("-w", "--whiten", help="Set the 'whiten' value to this", type=check_uint8, default=255)
	parser.add_argument("-o", "--output", help="Path to the output file", default="analog_result.jpg")

	return parser.parse_args()


def create_gui(img, *, min_thresh, max_thresh):
	cv2.imshow("Test", img)

	cv2.createTrackbar("blacken", "Test", 0, 255, slider(img))
	cv2.createTrackbar("whiten", "Test", 0, 255, slider(img))

	cv2.setTrackbarPos("whiten", "Test", max_thresh)
	cv2.setTrackbarPos("blacken", "Test", min_thresh)
	cv2.waitKey(0)

	return get_trackbar_values()


def slider(img):
	def update_img(_):
		min_thresh, max_thresh = get_trackbar_values()

		cpy = analoguify(img, max_thresh=max_thresh, min_thresh=min_thresh)
		cv2.imshow("Test", cpy)

	return update_img


def get_trackbar_values():
	min_thresh = cv2.getTrackbarPos("blacken", "Test")
	max_thresh = cv2.getTrackbarPos("whiten", "Test")

	return min_thresh, max_thresh


def analoguify(img, *, min_thresh, max_thresh):
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