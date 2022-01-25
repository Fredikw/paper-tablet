import numpy as np
import cv2
# Sort corners by id
# id and courner correspond on index
def sort_markers(ids, corners):
	ids = ids.tolist()
	lst = []
	for count, _ in enumerate(ids):
		idx = ids.index([count])
		lst.append(corners[idx])
	return tuple(lst)

# Input array of arrays
# Returns single array
def format_detectMarkers_corners(corners):

    entries = 32
    dim     = 2

    arr = np.empty([entries, dim], dtype="float32")
    arr_idx = 0
    for mark in corners:
        for corner in mark[0]:
            corner = np.array([corner])
            arr[arr_idx] = corner
            arr_idx += 1
    return arr


def save_image(img_name, img, path):
    cv2.imwrite(os.path.join(path , "tf_" + img_name), img)