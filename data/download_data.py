import kagglehub

path_dataset = kagglehub.dataset_download("ambityga/imagenet100")
print("Path to dataset files:", path_dataset)

path_labels = kagglehub.dataset_download("juliangarratt/imagenet-class-index-json")
print("Path to labels dictionary:", path_labels)