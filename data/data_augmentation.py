import pandas as pd
import os
import pretty_midi as pm
import ast

def data_augmentation_time(file_path):
    return 0

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

if __name__ == "__main__":
    dataset = "/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_tsv"
    folder_name = "/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset_augmentation_pitch"
    store_to_tsv_aug_pitch(dataset, folder_name)