import numpy as np
import os
from PIL import Image
import pandas as pd


parent_folder = "/home/barrage/grp3/"

csv_col = pd.read_csv(parent_folder+"crops/raw_crops/labels.csv")

distinct_projects = np.unique(csv_col["project"].tolist())


print(csv_col)
print(distinct_projects)

train_projects = ['TIDM_URBA/DIJON2021_1/3', 'TIDM_URBA/DIJON2021_2/3',  'TIDM_URBA/DIJON2021_3/3', 'TIDM_URBA/DIJON2021_1/4', 'VADEBIO', 'VADEBIO 2010', 'Pompey_1', 'RMQS']
test_projects = ['Parcelles Lysimétriques Sept 2007-1', 'Parcelles Lysimétriques Sept 2007-2', 'Parcelles Lysimétriques Sept 2007/2008-3', 'Bioindicateur', 'Bioindicateur 2 2009/2010', 'Pompey_1 2007', 'RMQS Biodiv Printemps 2006']

for p in distinct_projects :
    if p not in test_projects and p not in train_projects :
        train_projects.append(p)

print(train_projects)

train_dict = {"label1":[], "label2":[], "label3":[], "label4":[], "project": [], "img_name":[], "id":[]}
validation_dict = {"label1":[], "label2":[], "label3":[], "label4":[], "project": [], "img_name":[], "id":[]}


for i in range(len(csv_col)) :
    if csv_col["project"][i] in train_projects :
        #print("ok")
        train_dict["label1"].append(int(csv_col["label1"][i]))
        train_dict["label2"].append(int(csv_col["label2"][i]))
        train_dict["label3"].append(int(csv_col["label3"][i]))
        train_dict["label4"].append(int(csv_col["label4"][i]))
        train_dict["project"].append(csv_col["project"][i])
        train_dict["img_name"].append(csv_col["img_name"][i])
        train_dict["id"].append(csv_col["id"][i])

    else :
        validation_dict["label1"].append(int(csv_col["label1"][i]))
        validation_dict["label2"].append(int(csv_col["label2"][i]))
        validation_dict["label3"].append(int(csv_col["label3"][i]))
        validation_dict["label4"].append(int(csv_col["label4"][i]))
        validation_dict["project"].append(csv_col["project"][i])
        validation_dict["img_name"].append(csv_col["img_name"][i])
        validation_dict["id"].append(csv_col["id"][i])




txt_files = [f for f in os.listdir(parent_folder+"crops/background_patches") if f.endswith('.txt')]
for file in txt_files :
        file_path = os.path.join(parent_folder, "crops/background_patches", file)
        with open(file_path, "r") as f:
            values = f.read().strip().split()
        print(values)
        project_name = ""
        for k in range(1, len(values)):  # De 1 à len() - 4 pour ignorer les dernières valeurs
            project_name += values[k] + " "  
        print(project_name)

        if project_name in train_projects :
            train_dict["label1"].append(8)
            train_dict["label2"].append(8)
            train_dict["label3"].append(8)
            train_dict["label4"].append(8)
            train_dict["project"].append(project_name)
            train_dict["img_name"].append("../background_patches/"+file[:-4]+".jpg")
            train_dict["id"].append(file[:-4])

        else :
            validation_dict["label1"].append(8)
            validation_dict["label2"].append(8)
            validation_dict["label3"].append(8)
            validation_dict["label4"].append(8)
            validation_dict["project"].append(project_name)
            validation_dict["img_name"].append("../background_patches/"+file[:-4]+".jpg")
            validation_dict["id"].append(file[:-4])













train_df = pd.DataFrame(train_dict)
train_df.to_csv(parent_folder+"crops/raw_crops/train_labels.csv", index=False)
validation_df = pd.DataFrame(validation_dict)
validation_df.to_csv(parent_folder+"crops/raw_crops/validation_labels.csv", index=False)






























