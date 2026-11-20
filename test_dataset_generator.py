from data.dataset_generator import DatasetGenerator

generator = DatasetGenerator(seed=42)

df = generator.generate(
    num_graphs=100,
    partitions=4
)

print(df.head())

print("\nShape")
print(df.shape)

print("\nClass Distribution")
print(df["label"].value_counts())