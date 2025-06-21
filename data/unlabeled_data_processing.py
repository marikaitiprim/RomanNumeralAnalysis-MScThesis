import os
import pretty_midi as pm
import pandas as pd
from collections import defaultdict
import numpy as np

def get_key_at_time(key_changes, t):
    """Returns the key name at time t based on key signature changes."""
    if not key_changes:
        return None
    applicable_changes = [kc for kc in key_changes if kc.time <= t]
    if not applicable_changes:
        return None
    current_key = applicable_changes[-1]
    return pm.key_number_to_key_name(current_key.key_number)


def extract_midi_info(file_path):
    """
    Extracts MIDI information from a specific .mid file.

    args: 
        file_path (str): Path to the MIDI file.

    returns: 
        pd.DataFrame: A DataFrame containing the extracted midi info.
    """
    midi_stream = pm.PrettyMIDI(file_path) #load the MIDI file with pretty_midi
    tempos, _ = midi_stream.get_tempo_changes() #get the tempo changes

    if len(tempos) == 0 or tempos[0] <= 0: # Check if tempos is empty or contains 0.0
        tempo = 120.0  # Default tempo
    else:
        tempo = tempos[0]  # Or pick another index if needed
    
    total_duration = midi_stream.get_end_time() #total duration of the MIDI file in seconds
    time_signatures = midi_stream.time_signature_changes, #time signature changes in the MIDI file 
    if len(midi_stream.time_signature_changes) > 0:
        time_signature = time_signatures[0]

    time_step = 60.0 / tempo / (8 / 4) #assuming 8 divisions per beat as in the original paper 
    time_grid = np.arange(0, total_duration + time_step, time_step)

    key_changes = midi_stream.key_signature_changes

    rows = [] #store the rows for the dataframe
    active_notes = defaultdict(float) #store the active notes at each time step
    note_events = [] #store the notes from MIDI
    for instrument in midi_stream.instruments:
        if not instrument.is_drum:  #exclude drum instruments
            for note in instrument.notes:
                note_events.append({
                "start": note.start,
                "end": note.end,
                "pitch": pm.note_number_to_name(note.pitch)
            })
                

    for t in time_grid:
        beat = t / (60.0 / tempo) # calculate the beat number at time t
        measure = int(beat) // 4
        sounding = []

        for note in note_events:
            if abs(note["start"] - t) < 1e-4:
                active_notes[note["pitch"]] = note["end"]
            if note["start"] <= t < note["end"]:
                sounding.append(note["pitch"])

        active_notes = {p: e for p, e in active_notes.items() if e > t}
        onset_notes = [n["pitch"] for n in note_events if abs(n["start"] - t) < 1e-4]
        is_onset = len(onset_notes) > 0
        duration_beats = time_step / (60.0 / tempo)

        local_key = get_key_at_time(key_changes, t)


        row = {
            'j_offset': round(beat, 6),
            's_duration': round(duration_beats, 6),
            's_measure': measure,
            's_notes': sounding,
            's_isOnset': is_onset,
            'a_localKey': local_key
    
        }
        rows.append(row)

    return pd.DataFrame(rows)

def store_to_tsv(dataset):

    for file in os.listdir(dataset): #browse through the files in the dataset directory

        if file.endswith('.mid'):
            file_path = os.path.join(dataset, file) #create the full path to the file

            print("Processing file:", file)
            df_midi_info = extract_midi_info(file_path) #extract MIDI info

            tsv_filename = file.replace('.mid', '.tsv')
            df_midi_info.to_csv(tsv_filename, sep='\t', index=False) #store to .tsv file


if __name__ == "__main__":
    dataset = "/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset"
    store_to_tsv(dataset)