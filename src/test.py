import facenet


import cv2
import imutils


def get_image_paths_and_labels(dataset):
    mapping_label2ImgName = []
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
        mapping_label2ImgName.append(dataset[i].name)
    return image_paths_flat, labels_flat, mapping_label2ImgName


dataset = facenet.get_dataset('../videos/result')

print(dataset)

print(len(dataset))

print(dataset[0].name)
print(dataset[0].image_paths)
print(len(dataset[0].image_paths))


image_paths_flat, labels_flat, mapping_label2ImgName = get_image_paths_and_labels(dataset)

print('image_paths_flat: ', image_paths_flat)
print('labels_flat: ', labels_flat)
print('mapping_label2ImgName: ', mapping_label2ImgName)




