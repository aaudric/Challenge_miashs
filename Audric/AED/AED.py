import pandas as pd

# 1) Lire le CSV
csv_path = "../../barrage/grp3/crops/raw_crops/labels.csv"
df = pd.read_csv(csv_path)

# 2) Sélectionner les quatre colonnes de labels
label_cols = ["label1", "label2", "label3", "label4"]
labels_df = df[label_cols]

# 3) Aplatir tous les labels en un seul tableau
#    (on transforme le DataFrame [nb_lignes x 4 colonnes] en un tableau 1D)
all_labels = labels_df.values.flatten()

# 4) Compter le nombre d'occurrences de chaque label de 0 à 8
#    Méthode 1 : via pandas.Series.value_counts()
counts_series = pd.Series(all_labels).value_counts()

print("Nombre total d'occurrences par label (via value_counts) :")
for label in sorted(counts_series.index):
    print(f"Label {label} : {counts_series[label]}")
    