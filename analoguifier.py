import cv2
import numpy as np
import time



def main():
	img = load_image("test_image.jpg")
	img = cv2.resize(img, (800, 600))
	cv2.imshow("Test", img)

	cv2.createTrackbar("blacken", "Test", 0, 255, slider(img))
	cv2.createTrackbar("whiten", "Test", 0, 255, slider(img))
	cv2.setTrackbarPos("whiten", "Test", 255)
	cv2.waitKey(0)


def apply_filters(img, min_thresh, max_thresh):
	cpy = img.copy()
	cpy = blacken_dark_pixels(cpy, min_thresh)
	cpy = whiten_bright_pixels(cpy, max_thresh)
	cpy = normalize(cpy, max_thresh, min_thresh)

	return cpy


def slider(img):
	def update_img(_):
		min_thresh = cv2.getTrackbarPos("blacken", "Test")
		max_thresh = cv2.getTrackbarPos("whiten", "Test")

		cpy = apply_filters(img, max_thresh=max_thresh, min_thresh=min_thresh)
		cv2.imshow("Test", cpy)

	return update_img


def load_image(path: str):
	img = cv2.imread(path)
	return img


def blacken_dark_pixels(img, threshold: int = 50):
	cpy = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = gray <= threshold
	cpy[mask] = [0, 0, 0]

	return cpy


def whiten_bright_pixels(img, threshold: int = 230):
	cpy = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = gray >= threshold
	cpy[mask] = [255, 255, 255]

	return cpy


def normalize(img, max_thresh, min_thresh):
	cpy = img.copy().astype(np.float32)

    # Normalize each channel independently
	cpy = (cpy - min_thresh) / max(max_thresh - min_thresh, 1)
	cpy = np.clip(cpy, 0, 1) * 255

	return cpy.astype(np.uint8)


def store_image(img):
	cv2.imwrite("result.jpg", img)


if __name__ == '__main__':
	main()