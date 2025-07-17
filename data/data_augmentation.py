import pandas as pd
import os
import pretty_midi as pm
import ast

def data_augmentation_time(file_path, factor=1.2):
    """"
    Performs data augmentation on the durations of the notes in the MIDI file.
    args: 
        file_path (str): Path to the MIDI file.
        factor (float): Factor by which to multiply the j_offset and s_duration values to accelerate (if factor < 1) or slow down (if factor > 1). 
                        Default is 1.2. 
    returns:  
        pd.DataFrame: A DataFrame containing the extracted midi info.                
    """
    df = pd.read_csv(file_path, sep='\t') # Read the tsv file into a DataFrame
    df['j_offset'] = round(df['j_offset'].astype(float) * factor, 6)
    df['s_duration'] = round(df['s_duration'].astype(float) * factor, 6)

    beats_per_measure = 4 #assuming 4 beats per measure as in the original dataset construction
    df["s_measure"] = (df["j_offset"] // beats_per_measure).astype(int) + 1

    return df 

def split_notes(x):
    try:
        # Convert string like "['C#5, G#4']" → ['C#5, G#4']
        parsed = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
            return [n.strip() for n in parsed[0].split(',')]
        return parsed
    except Exception:
        return x  # fallback if x is malformed

def data_augmentation_pitch(file_path):
    """"
    Performs data augmentation on the pitch of the MIDI file.
    args: 
        file_path (str): Path to the MIDI file.
    returns:  
        pd.DataFrame: A DataFrame containing the extracted midi info.                
    """
    df = pd.read_csv(file_path, sep='\t') # Read the tsv file into a DataFrame
    df['s_notes'] = df['s_notes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() not in ['', '[]'] else []) #convert from string to list
    df['s_notes'] = df['s_notes'].apply(lambda x: [note.strip() for note in x if note.strip()]) #remove whitespaces
    df['s_notes'] = df['s_notes'].apply(lambda x: [pm.note_number_to_name(pm.note_name_to_number(n) + 1) for n in x] if x else []) # increase pitch by one semitone
    df['s_notes'] = df['s_notes'].apply(lambda x: [', '.join(x)] if x else '') #convert back to string

    df['s_notes'] = df['s_notes'].apply(split_notes)

    return df   

def store_to_tsv_aug_pitch(dataset, folder_name):

    for file in os.listdir(dataset): #browse through the files in the dataset directory

        if file.endswith('.tsv'):
            file_path = os.path.join(dataset, file) #create the full path to the file

            print("Processing file:", file)
            df_midi_info = data_augmentation_pitch(file_path) 

            tsv_filename = file.replace('.tsv', '_augmented_pitch.tsv') #create the new filename for the augmented data
            tsv_filename = os.path.join(folder_name, tsv_filename)
            df_midi_info.to_csv(tsv_filename, sep='\t', index=False) #store to .tsv file

def store_to_tsv_aug_time(dataset, folder_name):

    for file in os.listdir(dataset): #browse through the files in the dataset directory

        if file.endswith('.tsv'):
            file_path = os.path.join(dataset, file) #create the full path to the file

            print("Processing file:", file)
            df_midi_info = data_augmentation_time(file_path) 

            tsv_filename = file.replace('.tsv', '_augmented_time.tsv') #create the new filename for the augmented data
            tsv_filename = os.path.join(folder_name, tsv_filename)
            df_midi_info.to_csv(tsv_filename, sep='\t', index=False) #store to .tsv file

if __name__ == "__main__":
    dataset = "/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_tsv"
    folder_name_pitch = "/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_augmentation_pitch"
    folder_name_time = "/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_augmentation_time"
    store_to_tsv_aug_pitch(dataset, folder_name_pitch)
    store_to_tsv_aug_time(dataset, folder_name_time)