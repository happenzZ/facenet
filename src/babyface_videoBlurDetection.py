import cv2
import imutils
from scipy import misc
import os


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    #  measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


print('cwd: ', os.getcwd())


videoName = 'IMG_7378'
videoPath = '../videos/raw/{}.MOV'.format(videoName)
savePath = '../videos/blur/{}'.format(videoName)

# read video
capture = cv2.VideoCapture(videoPath)

if not capture.isOpened():
    print("could not open :", videoPath)
else:
    totalFrame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)

    videoFormat = capture.get(cv2.CAP_PROP_FORMAT)

    print('totalFrame: ', totalFrame)
    print('width: ', width)
    print('height: ', height)
    print('fps: ', fps)
    print('videoFormat: ', videoFormat)

frameIdx = -1
while True:
    frameIdx += 1
    print('frameIdx: ', frameIdx)
    # grab the current frame
    (grabbed, frame) = capture.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if not grabbed:
        break

    if frameIdx % 10 != 0:
        continue

    if frameIdx > 400:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fm = variance_of_laplacian(gray)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'blur: {}'.format(fm), (50, 50), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
    misc.imsave(savePath+'/{}.png'.format(frameIdx), frame)

