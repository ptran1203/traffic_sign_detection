import cv2


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def get_img(path, width=None):
    img = cv2.imread(path)
    if width:
        img = image_resize(img, width=width)

    return img


def draw_bbox(img, coordinates, text="face", color=(158, 0, 148)):
    "The pixcel's range should be [0, 255]"
    x, y, w, h = coordinates
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    # cv2.rectangle(img, (x, y), (x + w, y - 25), color, -1)
    return cv2.putText(img, text, (x + w, y - 10), 0, 0.5, (255, 255, 255))
