import cv2
import argparse
import moviepy

import numpy as np

from scipy.special import expit


def main():
	args = get_cmd_args()

	if args.video:
		img = get_video_frame(args.file)
	else:
		img = cv2.imread(args.file)

	smal = cv2.resize(img, tuple(int(s*args.scale) for s in img.shape[:2])[::-1])

	min_thresh, max_thresh = args.blacken, args.whiten
	balance, kernel = args.balance, args.kernel

	if args.gui:
		min_thresh, max_thresh, balance, kernel = create_gui(smal, min_thresh=min_thresh, max_thresh=max_thresh, balance=balance, kernel=kernel)

	if args.video:
		process_video_file(args.file, max_thresh, min_thresh, balance, kernel, args.output)
	else:
		process_img_file(img, max_thresh, min_thresh, balance, kernel, args.output)


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
	parser.add_argument("-b", "--blacken", help="Set the 'blacken' value to this", type=check_uint8, default=20)
	parser.add_argument("-w", "--whiten", help="Set the 'whiten' value to this", type=check_uint8, default=235)
	parser.add_argument("-ba", "--balance", help="Set the 'balance' value to this", type=np.uint8, default=25)
	parser.add_argument("-k", "--kernel", help="Set the 'kernel' value to this", type=np.uint8, default=100)
	parser.add_argument("-o", "--output", help="Path to the output file")
	parser.add_argument("-v", "--video", help="File is a video.", action="store_true")

	args = parser.parse_args()

	if not args.output:
		args.output = "analog_result." + args.file.split(".")[-1]

	return args


def get_video_frame(filepath):
	video = moviepy.VideoFileClip(filepath)
	frame = video.get_frame(0)
	bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

	return bgr_frame


def create_gui(img, *, min_thresh, max_thresh, balance, kernel):
	cv2.imshow("Test", img)

	cv2.createTrackbar("blacken", "Test", min_thresh, 255, slider(img))
	cv2.createTrackbar("whiten", "Test", max_thresh, 255, slider(img))
	cv2.createTrackbar("balance", "Test", balance, 100, slider(img))
	cv2.createTrackbar("kernel", "Test", kernel, 200, slider(img))

	cv2.waitKey(0)

	return get_trackbar_values()


def slider(img):
	def update_img(_):
		min_thresh, max_thresh, balance, kernel = get_trackbar_values()

		cpy = analoguify(img, max_thresh=max_thresh, min_thresh=min_thresh, balance=balance, kernel=kernel)
		cv2.imshow("Test", cpy)

	return update_img


def get_trackbar_values():
	min_thresh = cv2.getTrackbarPos("blacken", "Test")
	max_thresh = cv2.getTrackbarPos("whiten", "Test")
	balance = cv2.getTrackbarPos("balance", "Test")
	kernel = cv2.getTrackbarPos("kernel", "Test")

	return min_thresh, max_thresh, balance, kernel


def process_img_file(img, max_thresh, min_thresh, balance, kernel, output):
	result = analoguify(img, max_thresh=max_thresh, min_thresh=min_thresh, balance=balance, kernel=kernel)
	cv2.imwrite(output, result)


def process_video_file(filepath, max_thresh, min_thresh, balance, kernel, output):
	video = moviepy.VideoFileClip(filepath)

	edited = video.image_transform(lambda frame: analoguify(frame, max_thresh=max_thresh, min_thresh=min_thresh, balance=balance, kernel=kernel))

	edited.write_videofile(output)


def analoguify(img, *, min_thresh, max_thresh, balance, kernel):
	rad = radicalize(img, min_thresh=min_thresh, max_thresh=max_thresh)
	nrm = normalize(rad, max_thresh, min_thresh)
	blr = blur(nrm, balance, kernel)

	return blr


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
	cpy = smooth(np.clip(cpy, 0, 1)) * 255

	return cpy.astype(np.uint8)


def smooth(x):
	return 3*x**2 - 2*x**3


def blur(img, balance, kernelsize):
	if kernelsize <= 0:
		return img.copy()
	if kernelsize % 2 == 0:
		kernelsize += 1

	balance /= 100

	blurred = cv2.GaussianBlur(img.copy(), (kernelsize, kernelsize), 0)

	result = blurred*balance + img*(1-balance)
	result = result.clip(0, 255)

	return result.astype(np.uint8)


if __name__ == '__main__':
	main()
