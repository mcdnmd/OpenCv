import cv2
import numpy as np

FILE_PATH = ''


def create_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([50, 50, 50])
    upper_bound = np.array([150, 255, 255])
    return cv2.inRange(hsv, lower_bound, upper_bound)


def draw_contours(contours, frame, mask):
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 0, 0))


def show(frame, mask):
    mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    one_image = np.hstack((frame, mask_BGR))
    cv2.imshow('View', one_image)


def main():
    if FILE_PATH != '':
        cap = cv2.VideoCapture(FILE_PATH)
    else:
        cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if success:
            mask = create_mask(frame)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                draw_contours(contours, frame, mask)

            show(frame, mask)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
