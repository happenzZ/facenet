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
import random
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from src.faceExpression.model_face_zhp import build_model

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess = tf.Session()
    print('sess: ', sess)
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, './align/')

print('Creating MTCNN network for face detection')
network = importlib.import_module('models.inception_resnet_v1', 'inference')

print('Creating networks for face expression')
faceExpressionModel = build_model(4, 'CustomModelName')
faceExpressionModel.restore()

# data = np.array([])
# imagePath1 = '/Users/zhp/Project/facenet/faceDataset/test/1/0.png'
# imagePath2 = '/Users/zhp/Project/facenet/faceDataset/test/6/0.png'
# image1 = cv2.imread(imagePath1)
# image2 = cv2.imread(imagePath2)
#
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# reimage1 = cv2.resize(image1, (48, 48))
# reimage2 = cv2.resize(image2, (48, 48))
#
#
# data = reimage1
# data = np.vstack((data, reimage2))
#
# data = data.astype(np.float32) / 255 - 0.5
# print('data.shape: ', data.shape)
#
# data = np.array(data, dtype=np.uint8).reshape((-1, 48, 48, 1))
#
# print('data.shape: ', data.shape)
#
# logits, predictions = faceExpressionModel.infer_in_batches(faceExpressionModel.sess, data, batch_size=data.shape[0])
#
# print('logits: ', logits)
# print('predictions: ', predictions)



# variable setting
videoName = 'IMG_7378'
videoID = '000001'
kindergartenName = 'Matsubara'
kindergartenID = '000001'
frameInterval = 10


facePath = '../faceDataset/people_IMG_7378'
videoPath = '../videos/raw/{}.MOV'.format(videoName)
n_neighbors = 5
distance_threshold = 0.4
blur_threshold = 100
emotion_category = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def emotion_mapping(emotion_prediction):
    emotion_prob = {}
    for idx, category in enumerate(emotion_category):
        emotion_prob[category] = float(emotion_prediction[idx])
    return emotion_prob


def face_expression(images):
    # get face expression
    images = np.array(images, dtype=np.uint8).reshape((-1, 48, 48, 1))
    images = images.astype(np.float32) / 255 - 0.5
    _, emotion_predictions = faceExpressionModel.infer_in_batches(faceExpressionModel.sess, images,
                                                                  batch_size=images.shape[0])
    return emotion_predictions


def processing_image(image_paths):
    prewhitened_images = []
    for image_path in image_paths:
        image = misc.imread(image_path)
        prewhitened = facenet.prewhiten(image)
        prewhitened_images.append(prewhitened)
    return prewhitened_images


def get_dataset(path, sess, images_placeholder, embeddings):
    print('sess: ', sess)
    print('images_placeholder: ', images_placeholder)
    print('embeddings: ', embeddings)

    image_embeddings = np.array([])
    image_labels = np.array([])
    path_exp = os.path.expanduser(path)
    classes = [x for x in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, x))]
    classes.remove('known')
    classes.remove('unknown')
    print('classes: ', classes)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        if os.path.isdir(facedir):
            images = [x for x in os.listdir(facedir) if x.endswith('.png')]
            image_paths = [os.path.join(facedir, img) for img in images]
            prewhitened_images = processing_image(image_paths)

            feed_dict = {images_placeholder: prewhitened_images}
            tmp_embeddings = sess.run(embeddings, feed_dict=feed_dict)

            if image_embeddings.shape[0] == 0:
                image_embeddings = tmp_embeddings
            else:
                image_embeddings = np.vstack((image_embeddings, tmp_embeddings))
            image_labels = np.hstack((image_labels, [class_name] * tmp_embeddings.shape[0]))
    return image_embeddings, image_labels


def get_people(path):
    classes = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    classes.remove('known')
    classes.remove('unknown')
    return classes


def knn_judge(dist, ind, image_labels):
    label_distIdx = {}
    label_count = []
    label_list = []  # just record label
    for idx in range(ind.shape[0]):
        label = image_labels[ind[idx]]
        if label not in label_distIdx:
            label_distIdx[label] = []
        label_distIdx[label].append(idx)
        if label not in label_list:
            label_list.append(label)
    for label in label_list:
        label_count.append([label, len(label_distIdx[label])])

    label = sorted(label_count, key=lambda x: x[1], reverse=True)[0][0]
    distance = np.mean(dist[label_distIdx[label]])
    return distance, label


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
    grayed_images = []
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
            grayed = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
            grayed = cv2.resize(grayed, (48, 48))
            prewhitened = facenet.prewhiten(aligned)
            crop_images.append(aligned)
            prewhitened_images.append(prewhitened)
            grayed_images.append(grayed)
    return crop_images, prewhitened_images, grayed_images


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
    # mongodb
    client = MongoClient('localhost', 27017)
    db = client['babyface']
    # collections
    video_info = db['video_info']
    people_info = db['people_info']
    people = db['people']

    # with tf.Graph().as_default():
    tf.set_random_seed(args.seed)
    g = tf.Graph()
    with g.as_default():

        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3),
                                            name='input')

        # Build the inference graph
        prelogits, _ = network.inference(images_placeholder, 1.0,
                                         phase_train=False, weight_decay=0.0)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        sess = tf.Session()
        #    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        #    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        with sess.as_default():
            saver = tf.train.Saver()
            print(tf.train.latest_checkpoint('asset/train'))
            saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

            # get database image embeddings and labels
            image_embeddings, image_labels = get_dataset(facePath, sess, images_placeholder, embeddings)

            print('image_embeddings.shape', image_embeddings.shape)
            print('image_labels.shape', image_labels.shape)

            # mongodb create people collection doc
            people_video = {}

            # mongodb create people info
            people_list = get_people(facePath)
            for idx in range(len(people_list)):
                people_info_doc = {
                    'personID': people_list[idx],
                    'personName': people_list[idx],
                    'age': random.randint(1, 10),
                    'height': random.uniform(80, 120),
                    'imageFolder': os.path.abspath(os.path.join(facePath, people_list[idx]))
                }
                people_info.insert_one(people_info_doc)

                people_video[people_list[idx]] = {
                    'personID': people_list[idx],
                    'personName': people_list[idx],
                    'videoID': videoID,
                    'videoName': videoName,
                    'videoURL': os.path.abspath(videoPath),
                    'record': []
                }

            # mongodb create kindergarten_1_video_1 collection
            kdgt_video = '{}_{}'.format(kindergartenName, videoName)

            kdgt_video_coll = db[kdgt_video]

            # read video
            capture = cv2.VideoCapture(videoPath)

            if not capture.isOpened():
                print("could not open :", videoPath)
                return
            else:
                totalFrame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = capture.get(cv2.CAP_PROP_FPS)

                videoFormat = capture.get(cv2.CAP_PROP_FORMAT)

                print('totalFrame: ', totalFrame)
                print('width: ', frameWidth)
                print('height: ', frameHeight)
                print('fps: ', fps)
                print('videoFormat: ', videoFormat)

            frameNum = 0
            frameIdx = -1
            while True:
                frameIdx += 1
                # grab the current frame
                (grabbed, frame) = capture.read()

                # if we are viewing a video and we did not grab a frame,
                # then we have reached the end of the video
                if not grabbed:
                    break

                if frameIdx % frameInterval != 0:
                    continue

                # if frameIdx > 400:
                #     break

                frameNum += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                fm = variance_of_laplacian(gray)
                print('fm: ', fm)
                # detect blur
                if fm < blur_threshold:
                    continue

                print('frameIdx: ', frameIdx)

                # start one frame
                oneFrameInfo = {
                    "frameIdx": frameIdx,
                    "faces": []
                }

                bounding_boxes = detect_face(frame, 0.9)
                crop_images, prewhitened_images, grayed_images = crop_face(frame, bounding_boxes, args.image_size,
                                                                           args.margin)
                print('len(crop_images): ', len(crop_images))

                feed_dict = {images_placeholder: prewhitened_images}
                print('Start calculating result')
                tmp_embeddings = sess.run(embeddings, feed_dict=feed_dict)
                print('img_embeddings.shape: ', tmp_embeddings.shape)

                emotion_predictions = face_expression(grayed_images)

                if image_embeddings.shape[0] == 0:
                    print('not image_embeddings, error!')
                    break

                # KNN find neighbors
                nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(image_embeddings)
                distances, indices = nbrs.kneighbors(tmp_embeddings)
                # print('distances: ', distances)
                # print('indices: ', indices)

                for idx in range(indices.shape[0]):
                    dist = distances[idx]
                    ind = indices[idx]

                    distance, label = knn_judge(dist, ind, image_labels)
                    # print('distance: ', distance)
                    # print('label: ', label)
                    if distance > distance_threshold:
                        savePath = facePath + '/{}'.format('unknown')
                    else:
                        savePath = facePath + '/{}'.format('known')

                    if not os.path.exists(savePath):
                        os.makedirs(savePath)
                    misc.imsave(savePath + '/{}_{}.png'.format(label, frameIdx), crop_images[idx])

                    faceInfo = {
                        'faceID': '{}'.format(label),
                        'faceName': '{}'.format(label),
                        'position': bounding_boxes[idx, 0:4].tolist(),
                        'emotion': emotion_mapping(emotion_predictions[idx])
                    }
                    oneFrameInfo['faces'].append(faceInfo)

                    if distance < distance_threshold:
                        people_video[label]['record'].append({
                            'frameID': frameIdx,
                            'position': bounding_boxes[idx, 0:4].tolist(),
                            'emotion': emotion_category[np.argmax(emotion_predictions[idx])]
                        })

                print('oneFrameInfo: ', oneFrameInfo)
                kdgt_video_coll.insert_one(oneFrameInfo)

                frame = imutils.resize(frame, width=600)
                cv2.imshow('frame', frame)
                ck = cv2.waitKey(30) & 0xFF
                if ck == 27:
                    break

            video_info_doc = {
                'kdgtName': kindergartenName,
                'kdgtID': kindergartenID,
                'videoName': videoName,
                'videoID': videoID,
                'videoURL': os.path.abspath(videoPath),
                'totalFrame': totalFrame,
                'frameWidth': frameWidth,
                'frameHeight': frameHeight,
                'fps': fps,
                'frameInterval': frameInterval,
                'frameNum': frameNum
            }

            video_info.insert_one(video_info_doc)

            # mongodb insert people document
            for idx in range(len(people_list)):
                people_doc = people_video[people_list[idx]]
                people.insert_one(people_doc)

            # print('image_embeddings.shape: ', image_embeddings.shape)
            # print('image_labels.shape: ', image_labels.shape)
            # print('oneFrameInfo: ', oneFrameInfo)
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
