# -*- coding: utf-8 -*-

import os.path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import sugartensor as tf
import numpy as np
import importlib
import argparse
import facenet
import align.detect_face
from six.moves import xrange
from scipy import misc
import cv2
import imutils
import json
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from matplotlib import pyplot as plt

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, './align/')

network = importlib.import_module('models.inception_resnet_v1', 'inference')

# variable setting
videoName = 'IMG_7378'
faceGenerationPath = '../videos/faceGeneration/{}'.format(videoName)
videoPath = '../videos/raw/{}.MOV'.format(videoName)

if not os.path.exists(faceGenerationPath):
    os.makedirs(faceGenerationPath)


def save_image(img, faceRectangles, imgPath):
    # # Draw a rectangle around the faces
    for faceIdx, (left, top, right, bottom, p) in enumerate(faceRectangles):
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'faceIdx: {}'.format(faceIdx), (int(left), int(top)), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    misc.imsave(imgPath, img)
    return


def detect_face(img, probThreshold):
    # detect face bounding boxes from each frame
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    bounding_boxes = np.array([item for item in filter(lambda x: x[4] > probThreshold, bounding_boxes)])
    return bounding_boxes


def crop_face(img, bounding_boxes, image_size, margin):
    # crop face from each frame based on bounding boxed detected
    img_size = np.asarray(img.shape)[0:2]
    nrof_faces = bounding_boxes.shape[0]
    crop_images = []
    prewhitened_images = []
    if nrof_faces > 0:
        for faceIdx in xrange(nrof_faces):
            # filter out low probability
            # if bounding_boxes[faceIdx][4] < 0.9:
            #     continue
            det = np.squeeze(bounding_boxes[faceIdx, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            crop_images.append(aligned)
            prewhitened_images.append(prewhitened)
    return crop_images, prewhitened_images


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    #  measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def get_image_paths_and_labels(dataset):
    mapping_label2ImgName = []
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
        mapping_label2ImgName.append(dataset[i].name)
    return image_paths_flat, labels_flat, mapping_label2ImgName


def process_frame(frame, image_size, margin):
    print('process_frame')
    bounding_boxes = detect_face(frame)
    crop_images = crop_face(frame, bounding_boxes, image_size, margin)
    print('len(crop_images): ', len(crop_images))
    return


def main(args):
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

        if frameIdx % 60 != 0:
            continue

        if frameIdx > 10000:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fm = variance_of_laplacian(gray)
        print('fm: ', fm)
        # detect blur
        if fm < 100:
            continue

        bounding_boxes = detect_face(frame, 0.9)
        crop_images, prewhitened_images = crop_face(frame, bounding_boxes, args.image_size, args.margin)
        print('len(crop_images): ', len(crop_images))

        for idx, cropImg in enumerate(crop_images):
            cropImgPath = faceGenerationPath + '/{}_{}.png'.format(frameIdx, idx)
            misc.imsave(cropImgPath, cropImg)

        frame = imutils.resize(frame, width=600)
        cv2.imshow('frame', frame)
        ck = cv2.waitKey(30) & 0xFF
        if ck == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
    print('hello world')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, help='Images to compare', default='../faceDataset/test/lfw_test.jpg')
    # parser.add_argument('--image_file', type=str, help='Images to compare', default='../faceDataset/test/babyImage.jpg')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.67)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='../preModels/20161116-234200/model-20161116-234200.ckpt-80000')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
