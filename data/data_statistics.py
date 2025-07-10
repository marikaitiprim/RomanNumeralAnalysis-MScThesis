import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def count_key_changes(df):
    local_keys = df["a_localKey"].dropna().tolist()
    if not local_keys:
        return 0

    # Count the number of times the key changes from one row to the next
    change_count = sum(
        1 for i in range(1, len(local_keys)) if local_keys[i] != local_keys[i - 1]
    )
    return change_count

def plot_normalized_inversion_by_degree(df):
    grouped = df.groupby(["degree_num", "a_inversion"]).size().reset_index(name="count") # Count how many times each inversion appears for each degree

    grouped["Proportion"] = grouped.groupby("degree_num")["count"].transform(lambda x: x / x.sum())     # Normalize within each degree

    grouped = grouped[grouped["degree_num"].isin(range(1, 8))] # filter to keep only degrees 1-7

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=grouped,
        x="degree_num",
        y="Proportion",
        hue="a_inversion",
        palette="Set2"
    )
    plt.title("Normalized Inversion Distribution by Degree")
    plt.xlabel("Degree")
    plt.ylabel("Proportion")
    plt.legend(title="Inversion")
    plt.tight_layout()
    plt.show()

def plot_hr_histogram(combined_df):

    hr_means = ( # Group by degree number and compute mean harmonic rhythm
        combined_df.groupby("degree_num")["a_harmonicRhythm"]
        .mean()
        .reindex(range(1, 8))  # Ensure degrees 1–7 are in order
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=hr_means, x="degree_num", y="a_harmonicRhythm", palette="viridis")
    plt.title("Mean Harmonic Rhythm per Degree")
    plt.xlabel("Degree")
    plt.ylabel("Mean Harmonic Rhythm")
    plt.tight_layout()
    plt.show()

def extract_degree_number(degree_str):
    # Extract numeric part of the degree (ignore accidentals)
    if pd.isna(degree_str):
        return None
    match = re.search(r'(\d+)', str(degree_str))
    if match:
        num = int(match.group(1))
        if 1 <= num <= 7:
            return num
    return None

def get_data_statistics(folder):

    count_files = 0
    count_rows = 0
    count_tonicized = 0
    count_key = 0
    all_hr = []
    all_inv = []

    for root, _, files in os.walk(folder):

        for file in files: #browse through the files in the dataset directory

            if file.endswith('.tsv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, sep='\t') # Read the tsv file into a DataFrame

                count_files += 1
                count_rows += len(df)
                count_tonicized += df["a_degree2"].notna().sum() # count non-null values in the degree_2 column
                key_changes = count_key_changes(df)
                if key_changes > 5: #more than 5 key changes in a file
                    count_key += 1


                df["degree_num"] = df["a_degree1"].apply(extract_degree_number)
                sub_df_hr = df[["degree_num", "a_harmonicRhythm"]].dropna()
                sub_df_inv = df[["degree_num", "a_inversion"]].dropna()
                all_hr.append(sub_df_hr)
                all_inv.append(sub_df_inv)


    print("Total number of files:", count_files)
    print('Total number of rows:', count_rows)
    print("Total number of tonicized chords:", count_tonicized)
    print("Total number of files with more than 5 key changes:", count_key)

    combined_hr = pd.concat(all_hr, ignore_index=True) # Combine all collected data for harmonic rhythm and degrees
    plot_hr_histogram(combined_hr)

    combined_inv = pd.concat(all_inv, ignore_index=True)
    plot_normalized_inversion_by_degree(combined_inv)


if __name__ == "__main__":
    data_folder =  "/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/augmentednet_dataset"
    get_data_statistics(data_folder)

