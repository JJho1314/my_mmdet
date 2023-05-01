from class_name import *

coco = []
for i, _ in enumerate(COCO_CLASSES):
    if(i not in coco_unseen_ids_train):
        coco.append(COCO_CLASSES[i])
    
print(coco)