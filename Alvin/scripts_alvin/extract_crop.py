import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

n = '4'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["OMP_NUM_THREADS"] = n
os.environ["OPENBLAS_NUM_THREADS"] = n
os.environ["MKL_NUM_THREADS"] = n
os.environ["NUMEXPR_NUM_THREADS"] = n
os.sched_setaffinity(0, {0, 1, 2, 3})






folder_path = "/home/barrage/grp3/data/"
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
#print(txt_files)
print(len(txt_files))

file_names = []
labels = []
project = []
x = []
y = []
h = []
w = []


for m in range(len(txt_files))  :

    with open(folder_path+txt_files[m], 'r', encoding='utf-8') as file:

        for line in file:
            values = line.strip().split()  # Séparer par espace
            file_names.append(txt_files[m])
            all_labels = values[0].strip().split("_")
            #print(all_labels)
            labels.append([float(all_labels[j]) for j in range(4)])
            project_name = ""
            for k in range(1, len(values) - 4):  # De 1 à len() - 4 pour ignorer les dernières valeurs
                project_name += values[k] + " "  
            print(project_name)
            project.append(project_name.strip())

            project.append(values[1])
            x.append(float(values[-4]))
            y.append(float(values[-3]))
            w.append(float(values[-2]))
            h.append(float(values[-1]))       





dimensions1 = []
dimensions2 = []




background_proba = 1/20


for i, txt_file in enumerate(file_names) :
    print(i)
    name = "crop_"+str(i+1)
    image_file = txt_file.replace('.txt', '.jpg')
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    image_np = np.array(image)

    img_height, img_width = image_np.shape[:2]

    xcentre = int(x[i]*img_width)
    ycentre = int(y[i]*img_height)
    largeur = int(w[i]*img_width)
    hauteur = int(h[i] *img_height)
    
    xmin = int(xcentre - largeur //2)
    xmax = int(xmin+largeur)
    ymin = int(ycentre - hauteur //2)
    ymax = int(ymin + hauteur)

    dimensions1.append(xmax-xmin)
    dimensions2.append(ymax-ymin)

    """
    #print(xmin, xmax, ymin, ymax, img_height, img_width)
    cropped_image = image_np[ymin:ymax, xmin:xmax]
    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        print(f"Image {name} est vide après découpe, saut de la sauvegarde.")
        continue


    cropped_image_pil = Image.fromarray(cropped_image)
    cropped_image_pil.save(os.path.join(image_save_folder, name + ".jpg"))


    

    labels_dict["label1"].append(labels[i][0])
    labels_dict["label2"].append(labels[i][1])
    labels_dict["label3"].append(labels[i][2])
    labels_dict["label4"].append(labels[i][3])
    labels_dict["project"].append(project[i])
    labels_dict["id"].append(txt_file[:-4])
    print(txt_file[:-4])
    labels_dict["img_name"].append(name+".jpg" )"""


dimensions2 = np.array(dimensions2)
dimensions1 = np.array(dimensions1)

#dimensions = np.concatenate([dimensions1, dimensions2], axis=1)  # N, 2

maxi = max(np.max(dimensions2), np.max(dimensions1)) + 10


template_boxes = [500, 1000, 1500, 2000,  maxi]




import random

labels_dict = {"label1":[], "label2":[], "label3":[], "label4":[], "project": [], "img_name":[], "id":[]}
image_save_folder = "/home/barrage/grp3/crops/fixed_crops/"

count_bg = 0

for i, txt_file in enumerate(file_names):
    print(i)
    name = "crop_" + str(i + 1)
    image_file = txt_file.replace('.txt', '.jpg')
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    image_np = np.array(image)

    img_height, img_width = image_np.shape[:2]

    xcentre = int(x[i] * img_width)
    ycentre = int(y[i] * img_height)
    largeur = int(w[i] * img_width)
    hauteur = int(h[i] * img_height)

    for tmp in template_boxes :
        if largeur < tmp :
            bounding_w = tmp
            break

    for tmp in template_boxes :
        if hauteur < tmp :
            bounding_h = tmp
            break


    



    xmin = int(xcentre - bounding_w // 2)
    xmax = int(xmin + bounding_w)
    ymin = int(ycentre - bounding_h // 2)
    ymax = int(ymin + bounding_h)


    cropped_image = image_np[ymin:ymax, xmin:xmax]
    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        print(f"Image {name} est vide après découpe, saut de la sauvegarde.")
        continue

    cropped_image_pil = Image.fromarray(cropped_image)
    cropped_image_pil.save(os.path.join(image_save_folder, name + ".jpg"))

    if np.any(np.array(labels[i]) == 8):
        print("j'ai trouvé un 8 !!")

    labels_dict["label1"].append(labels[i][0])
    labels_dict["label2"].append(labels[i][1])
    labels_dict["label3"].append(labels[i][2])
    labels_dict["label4"].append(labels[i][3])
    labels_dict["project"].append(project[i])
    labels_dict["id"].append(txt_file[:-4])
    labels_dict["img_name"].append(name + ".jpg")

    # Ajout du background aléatoire
    if random.random() < background_proba:
        # Tirer une taille de box aléatoire depuis dimensions
        random_idx = random.randint(0, len(dimensions1) - 1)
        inds = np.arange(len(template_boxes))
        sele = np.random.choice(inds, size=(2), replace=True)
        bg_height = dimensions2[sele[0]]
        bg_width = dimensions1[sele[1]]

        max_attempts = 10
        for _ in range(max_attempts):
            # Tirer un point de départ aléatoire
            bg_xmin = random.randint(0, img_width - bg_width)
            bg_ymin = random.randint(0, img_height - bg_height)
            bg_xmax = bg_xmin + bg_width
            bg_ymax = bg_ymin + bg_height

            # Calculer l'IoU avec l'objet d'intérêt
            inter_xmin = max(xmin, bg_xmin)
            inter_ymin = max(ymin, bg_ymin)
            inter_xmax = min(xmax, bg_xmax)
            inter_ymax = min(ymax, bg_ymax)

            inter_width = max(0, inter_xmax - inter_xmin)
            inter_height = max(0, inter_ymax - inter_ymin)
            inter_area = inter_width * inter_height

            obj_area = (xmax - xmin) * (ymax - ymin)
            bg_area = (bg_xmax - bg_xmin) * (bg_ymax - bg_ymin)
            iou = inter_area / (obj_area + bg_area - inter_area) if (obj_area + bg_area - inter_area) > 0 else 0

            if iou < 0.15:
                break
        else:
            print(f"Impossible de trouver un bon crop de fond pour {name}, skip.")
            continue

        # Extraire et sauvegarder le crop de background
        bg_crop = image_np[bg_ymin:bg_ymax, bg_xmin:bg_xmax]
        if bg_crop.shape[0] == 0 or bg_crop.shape[1] == 0:
            print(f"Image background pour {name} est vide après découpe, saut de la sauvegarde.")
            continue

        bg_name = f"background_{i+1}"
        bg_crop_pil = Image.fromarray(bg_crop)
        bg_crop_pil.save(os.path.join(image_save_folder, bg_name + ".jpg"))
        labels_dict["label1"].append(8)
        labels_dict["label2"].append(8)
        labels_dict["label3"].append(8)
        labels_dict["label4"].append(8)
        labels_dict["project"].append(project[i])
        labels_dict["id"].append(txt_file[:-4]+'_bg')
        labels_dict["img_name"].append(bg_name + ".jpg")
        print("j'ai pris un background !", count_bg+1)
        count_bg+=1








labels_df = pd.DataFrame(labels_dict)
labels_df.to_csv("/home/barrage/grp3/crops/fixed_crops/labels.csv", index=False)

toto()





import random

labels_dict = {"label1":[], "label2":[], "label3":[], "label4":[], "project": [], "img_name":[], "id":[]}
image_save_folder = "/home/barrage/grp3/crops/raw_crops/"

count_bg = 0

for i, txt_file in enumerate(file_names):
    print(i)
    name = "crop_" + str(i + 1)
    image_file = txt_file.replace('.txt', '.jpg')
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    image_np = np.array(image)

    img_height, img_width = image_np.shape[:2]

    xcentre = int(x[i] * img_width)
    ycentre = int(y[i] * img_height)
    largeur = int(w[i] * img_width)
    hauteur = int(h[i] * img_height)

    xmin = int(xcentre - largeur // 2)
    xmax = int(xmin + largeur)
    ymin = int(ycentre - hauteur // 2)
    ymax = int(ymin + hauteur)


    cropped_image = image_np[ymin:ymax, xmin:xmax]
    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        print(f"Image {name} est vide après découpe, saut de la sauvegarde.")
        continue

    cropped_image_pil = Image.fromarray(cropped_image)
    cropped_image_pil.save(os.path.join(image_save_folder, name + ".jpg"))

    if np.any(np.array(labels[i]) == 8):
        print("j'ai trouvé un 8 !!")

    labels_dict["label1"].append(labels[i][0])
    labels_dict["label2"].append(labels[i][1])
    labels_dict["label3"].append(labels[i][2])
    labels_dict["label4"].append(labels[i][3])
    labels_dict["project"].append(project[i])
    labels_dict["id"].append(txt_file[:-4])
    labels_dict["img_name"].append(name + ".jpg")

    # Ajout du background aléatoire
    if random.random() < background_proba:
        # Tirer une taille de box aléatoire depuis dimensions
        random_idx = random.randint(0, len(dimensions1) - 1)
        bg_height = dimensions2[random_idx]
        bg_width = dimensions1[random_idx]

        max_attempts = 10
        for _ in range(max_attempts):
            # Tirer un point de départ aléatoire
            bg_xmin = random.randint(0, img_width - bg_width)
            bg_ymin = random.randint(0, img_height - bg_height)
            bg_xmax = bg_xmin + bg_width
            bg_ymax = bg_ymin + bg_height

            # Calculer l'IoU avec l'objet d'intérêt
            inter_xmin = max(xmin, bg_xmin)
            inter_ymin = max(ymin, bg_ymin)
            inter_xmax = min(xmax, bg_xmax)
            inter_ymax = min(ymax, bg_ymax)

            inter_width = max(0, inter_xmax - inter_xmin)
            inter_height = max(0, inter_ymax - inter_ymin)
            inter_area = inter_width * inter_height

            obj_area = (xmax - xmin) * (ymax - ymin)
            bg_area = (bg_xmax - bg_xmin) * (bg_ymax - bg_ymin)
            iou = inter_area / (obj_area + bg_area - inter_area) if (obj_area + bg_area - inter_area) > 0 else 0

            if iou < 0.15:
                break
        else:
            print(f"Impossible de trouver un bon crop de fond pour {name}, skip.")
            continue

        # Extraire et sauvegarder le crop de background
        bg_crop = image_np[bg_ymin:bg_ymax, bg_xmin:bg_xmax]
        if bg_crop.shape[0] == 0 or bg_crop.shape[1] == 0:
            print(f"Image background pour {name} est vide après découpe, saut de la sauvegarde.")
            continue

        bg_name = f"background_{i+1}"
        bg_crop_pil = Image.fromarray(bg_crop)
        bg_crop_pil.save(os.path.join(image_save_folder, bg_name + ".jpg"))
        labels_dict["label1"].append(8)
        labels_dict["label2"].append(8)
        labels_dict["label3"].append(8)
        labels_dict["label4"].append(8)
        labels_dict["project"].append(project[i])
        labels_dict["id"].append(txt_file[:-4]+'_bg')
        labels_dict["img_name"].append(bg_name + ".jpg")
        print("j'ai pris un background !", count_bg+1)
        count_bg+=1








labels_df = pd.DataFrame(labels_dict)
labels_df.to_csv("/home/barrage/grp3/crops/raw_crops/labels.csv", index=False)

toto()



import matplotlib.pyplot as plt
plt.scatter(np.array(dimensions1), np.array(dimensions2))
plt.xlabel("largeur")
plt.ylabel("hauteur")
plt.title("dimensions des crop dans la base de train")
plt.savefig("train_dimension.png")



toto()





resized_crops = np.zeros((len(file_names), 448, 448, 3))
labels_dict = {"label1":[], "label2":[], "label3":[], "label4":[], "project": []}

for i, txt_file in enumerate(file_names) :
    print(i)
    image_file = txt_file.replace('.txt', '.jpg')
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    image_np = np.array(image)

    img_height, img_width = image_np.shape[:2]

    xcentre = int(x[i]*img_width)
    ycentre = int(y[i]*img_height)
    largeur = int(w[i]*img_width)
    hauteur = int(h[i] *img_height)
    
    xmin = int(xcentre - largeur //2)
    xmax = int(xmin+largeur)
    ymin = int(ycentre - hauteur //2)
    ymax = int(ymin + hauteur)


    
    print(xmin, xmax, ymin, ymax)
    resized_crops[i] = np.array(Image.fromarray(image_np[ymin:ymax, xmin:xmax]).resize((448, 448), Image.NEAREST))

    labels_dict["label1"].append(labels[i][0])
    labels_dict["label2"].append(labels[i][1])
    labels_dict["label3"].append(labels[i][2])
    labels_dict["label4"].append(labels[i][3])
    labels_dict["project"].append(project[i])


labels_dict["label1"] = np.array(labels_dict["label1"], dtype=np.int32)
labels_dict["label2"] = np.array(labels_dict["label2"], dtype=np.int32)
labels_dict["label3"] = np.array(labels_dict["label3"], dtype=np.int32)
labels_dict["label4"] = np.array(labels_dict["label4"], dtype=np.int32)
labels_dict["project"] = np.array(labels_dict["project"], dtype=str)

np.savez("../crops/croped_data_448.npz", images=resized_crops, labels=labels_dict)


fig, axes = plt.subplots(nrows=1, ncols=3)
for i in range(3) :
    axes[i].imshow(resized_crops[i])
    axes[i].axis("off")

fig.tight_layout()
fig.savefig("crop_plot_448.png")

toto()







project_counter = {}
class_counter = {}

for i in range(len(file_names)) :
    proj = project[i]
    if proj not in project_counter :
        project_counter[proj] = 1
    else :
        project_counter[proj] += 1


    classes = labels[i]
    for c in classes :
        if c not in class_counter :
            class_counter[c] = 1
        else :
            class_counter[c] += 1



proj_names_unique = [p for p in project_counter]
proj_names_val = [project_counter[p] for p in proj_names_unique]
plt.bar(proj_names_unique, proj_names_val)
plt.xlabel("projets")
plt.ylabel("nombre")
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.title("nombre de projet")
plt.savefig("train_project.png")
plt.close()

clas_names_unique = [str(int(p)) for p in class_counter]
clas_names_val = [class_counter[p] for p in class_counter]
plt.bar(clas_names_unique, clas_names_val)
plt.xlabel("classes")
plt.ylabel("nombre")
plt.title("nombre de labellisation par classe (somme sur les experts)")
plt.savefig("train_classes.png")








toto()

print(len(file_names))
dimensions1 = []
dimensions2 = []
print(len(file_names))


resized_crops = np.zeros((len(file_names), 224, 224, 3))
labels_dict = {"label1":[], "label2":[], "label3":[], "label4":[], "project": []}

for i, txt_file in enumerate(file_names) :
    print(i)
    image_file = txt_file.replace('.txt', '.jpg')
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    image_np = np.array(image)

    img_height, img_width = image_np.shape[:2]

    xmin = int(x[i]*img_width)
    xmax = int(xmin + int(w[i]*img_width))
    ymin = int(y[i] * img_height)
    ymax = int(ymin + int(h[i] * img_height))

    dimensions1.append(xmax - xmin)
    dimensions2.append(ymax - ymin)
    print(xmin, xmax, ymin, ymax)
    resized_crops[i] = np.array(Image.fromarray(image_np[xmin:xmax, ymin:ymax]).resize((224, 224), Image.NEAREST))

    labels_dict["label1"].append(labels[i][0])
    labels_dict["label2"].append(labels[i][1])
    labels_dict["label3"].append(labels[i][2])
    labels_dict["label4"].append(labels[i][3])
    labels_dict["project"].append(project[i])


labels_dict["label1"] = np.array(labels_dict["label1"], dtype=np.int32)
labels_dict["label2"] = np.array(labels_dict["label2"], dtype=np.int32)
labels_dict["label3"] = np.array(labels_dict["label3"], dtype=np.int32)
labels_dict["label4"] = np.array(labels_dict["label4"], dtype=np.int32)
labels_dict["project"] = np.array(labels_dict["project"], dtype=str)

np.savez("../crops/croped_data.npz", images=resized_crops, labels=labels_dict)



"""
import matplotlib.pyplot as plt
plt.scatter(np.array(dimensions1), np.array(dimensions2))
plt.xlabel("largeur")
plt.ylabel("hauteur")
plt.title("dimensions des crop dans la base de train")
plt.savefig("train_dimension.png")
"""

