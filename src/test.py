
import numpy as np

emotion_category = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
a = np.array([4, 6, 2.0, 9.5, 3.9])

print(np.argmax(a))
print(type(np.argmax(a)))

print(emotion_category[np.argmax(a)])
# import os
#
# facePath = '../faceDataset/people_IMG_7378'
#
# classes = [x for x in os.listdir(facePath) if os.path.isdir(os.path.join(facePath, x))]
# classes.remove('known')
# classes.remove('unknown')
# print('classes: ', classes)
#
#
# print(os.path.abspath(os.path.join(facePath, classes[0])))

# import numpy as np
#
# c = []
# a = np.array([[[1.0,2], [3,4]], [[5,6], [7,8]]])
#
# c = [a[0], a[1]]
#
#
# print('a: ', a)
# print('a.shape: ', a.shape)
#
# b = np.array(a, dtype=np.float32).reshape((-1, 2, 2, 1))
# print('b.shape: ', b.shape)
# print('b: ', b)
#
# e = np.array(c, dtype=np.uint8).reshape((-1, 2, 2, 1))
# print('e.shape: ', e.shape)
# print('e: ', e)

# from scipy import misc
# import cv2
# import numpy as np
#
# image = misc.imread('/Users/zhp/Project/facenet/faceDataset/test/0/0.png')
# # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
# aligned = misc.imresize(image, (80, 80), interp='bilinear')
#
# grayed = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
#
# cv2.imshow('name', grayed)
# ck = cv2.waitKey(0)
# cv2.destroyAllWindows()





#
#
# size = 10
# batch_size = 10
# for begin in range(0, size, batch_size):
#     print('begin: ', begin)
#     end = begin + batch_size
#     print('end: ', end)
#     if end > size:
#         end = size
#     print('end: ', end)








# import numpy as np
#
# dist = np.array([3.3, 5.2, 9.7])
# ind = np.array([2, 4, 1])
# image_labels = np.array(['a', 'b', 'b', 'd', 'e'])
#
#
# def knn_judge(dist, ind, image_labels):
#     label_distIdx = {}
#     label_count = []
#     label_list = []   # just record label
#     for idx in range(ind.shape[0]):
#         label = image_labels[ind[idx]]
#         if label not in label_distIdx:
#             label_distIdx[label] = []
#         label_distIdx[label].append(idx)
#         if label not in label_list:
#             label_list.append(label)
#     for label in label_list:
#         label_count.append([label, len(label_distIdx[label])])
#
#     print(sorted(label_count, key=lambda x: x[1], reverse=True))
#
#     print('label_distIdx: ', label_distIdx)
#     print('label_count: ', label_count)
#     print('label_list: ', label_list)
#
#     label = sorted(label_count, key=lambda x: x[1], reverse=True)[0][0]
#     distance = np.mean(dist[label_distIdx[label]])
#
#     print('label: ', label)
#     print(label_distIdx[label])
#     print(dist[label_distIdx[label]])
#     print('distance: ', distance)
#
# knn_judge(dist, ind, image_labels)

# from collections import Counter
# import numpy as np
# x = ['c','a','b','c','a','d']
#
#
# count = Counter(x)
# print(count)
#
# print(count.most_common())
#
#
# a = np.array(['a', 'b', 'a',  'b', 'c', 'c', 'b', 'c', 'a'])
# # a = ['a', 'b', 'a',  'b', 'c', 'c', 'b', 'c', 'a']
#
# b = [1, 2, 2, 3, 3, 3, 3, 3, 4, 5, 6, 6, 6]
#
# result = Counter(a.tolist()).most_common()
#
# print('result: ', result)
#
# print(result[0][0])
# print(result[1][0])
# print(Counter(b).most_common())
#
#
#
# c = np.array([0, 2, 1])
# d = np.array(['a', 'b', 'c'])
#
# e = d[c.tolist()]
#
# print(e)
# print(type(e))
#




# import cv2
# from scipy import misc
# import imutils
#
#
# def save_image(img, faceRectangles, imgPath):
#     # # Draw a rectangle around the faces
#     for faceIdx, (left, top, right, bottom, p) in enumerate(faceRectangles):
#         cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, 'faceIdx: {}'.format(faceIdx), (int(left), int(top)), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
#     misc.imsave(imgPath, img)
#     return
#
#
# def variance_of_laplacian(image):
#     # compute the Laplacian of the image and then return the focus
#     #  measure, which is simply the variance of the Laplacian
#     return cv2.Laplacian(image, cv2.CV_64F).var()
#
#
# # read video
# videoPath = '../videos/raw/IMG_7378.MOV'
# capture = cv2.VideoCapture(videoPath)
#
# if not capture.isOpened():
#     print("could not open :", videoPath)
# else:
#     totalFrame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
#     width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     fps = capture.get(cv2.CAP_PROP_FPS)
#
#     videoFormat = capture.get(cv2.CAP_PROP_FORMAT)
#
#     print('totalFrame: ', totalFrame)
#     print('width: ', width)
#     print('height: ', height)
#     print('fps: ', fps)
#     print('videoFormat: ', videoFormat)
#
# frameIdx = 0
# while True:
#     frameIdx += 1
#     print('frameIdx: ', frameIdx)
#     # grab the current frame
#     (grabbed, frame) = capture.read()
#
#     # if we are viewing a video and we did not grab a frame,
#     # then we have reached the end of the video
#     if not grabbed:
#         break
#
#     if frameIdx % 30 != 0:
#         continue
#
#     if frameIdx > 400:
#         break
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     fm = variance_of_laplacian(gray)
#
#     # detect blur
#     if fm < 400:
#         continue
#
#     frame = imutils.resize(frame, width=600)
#     cv2.imshow('frame', frame)
#     ck = cv2.waitKey(30) & 0xFF
#     if ck == 27:
#         break
#
#
# capture.release()
# cv2.destroyAllWindows()
# print('hello world')













# #####################################
#  Test mongodb
# #####################################
# from pymongo import MongoClient
#
# client = MongoClient('localhost', 27017)
#
# db = client['test']
# coll = db['kindergarten_1_video_1']

# result = coll.insert_one(
#         {
#             "frameID": 10,
#             "faces": [
#                 {
#                     "faceID": "12325",
#                     "faceName": "peter",
#                     "position": [10, 20, 30, 40],
#                     "emotion": {
#                         "angry": 0.4,
#                         "disgust": 0.3,
#                         "fear": 0.1,
#                         "happy": 0.9,
#                         "sad": 0.4,
#                         "surprise": 0.5,
#                         "neutral": 0.2
#                     }
#                 },
#                 {
#                     "faceID": "24234",
#                     "faceName": "Bob",
#                     "position": [50, 60, 70, 80],
#                     "emotion": {
#                         "angry": 0.4,
#                         "disgust": 0.3,
#                         "fear": 0.1,
#                         "happy": 0.9,
#                         "sad": 0.4,
#                         "surprise": 0.5,
#                         "neutral": 0.2
#                     }
#                 }
#             ]
#         }
# )

#
# cursor = coll.find()
# for document in cursor:
#     print(document)









# #####################################
#  Test facenet get_dataset
# #####################################
# import facenet
#
#
# import cv2
# import imutils
#
#
# def get_image_paths_and_labels(dataset):
#     mapping_label2ImgName = []
#     image_paths_flat = []
#     labels_flat = []
#     for i in range(len(dataset)):
#         image_paths_flat += dataset[i].image_paths
#         labels_flat += [i] * len(dataset[i].image_paths)
#         mapping_label2ImgName.append(dataset[i].name)
#     return image_paths_flat, labels_flat, mapping_label2ImgName
#
#
# dataset = facenet.get_dataset('../videos/result')
#
# print(dataset)
#
# print(len(dataset))
#
# print(dataset[0].name)
# print(dataset[0].image_paths)
# print(len(dataset[0].image_paths))
#
#
# image_paths_flat, labels_flat, mapping_label2ImgName = get_image_paths_and_labels(dataset)
#
# print('image_paths_flat: ', image_paths_flat)
# print('labels_flat: ', labels_flat)
# print('mapping_label2ImgName: ', mapping_label2ImgName)
