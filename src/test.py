# #####################################
#  Test mongodb
# #####################################
from pymongo import MongoClient

client = MongoClient('localhost', 27017)

db = client['test']
coll = db['kindergarten_1_video_1']

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


cursor = coll.find()
for document in cursor:
    print(document)









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
