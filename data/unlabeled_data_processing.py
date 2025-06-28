import os
import pretty_midi as pm
import pandas as pd
from collections import defaultdict

def get_offsets(midi_stream, step_size=0.125):
    """
    Returns the offsets for the given step size.
    args:
        midi_stream: PrettyMIDI object representing the MIDI file.
        step_size (float): Time step size in seconds. Default is 0.125 (1/8 note).
    """
    total_duration = midi_stream.get_end_time() #total duration of the MIDI file in seconds
    num_steps = int(total_duration // step_size) + 1
    j_offsets = [round(i * step_size, 6) for i in range(num_steps)] # divide the total duration into steps of step_size

    return j_offsets

def get_notes_onsets(note_events, j_offsets, step_size=0.125):
    """
    Determines the sounding notes and if a note is an onset at a given time step.
    args:
        note_events (list): List of note events with their start, end and pitch name.
        j_offsets (list): List of time offsets at which to check for onsets.
        step_size (float): Time step size in seconds. Default is 0.125 (1/8 note).
    returns:
        s_notes (list): List of lists containing the names of the sounding notes at each time step.
        s_isOnset (list): List of lists indicating whether each note is an onset at each time step (boolean values).
    """

    for note in note_events: 
        note["rounded_start"] = round(round(note["start"] / step_size) * step_size, 6) #round the start time to the nearest step size

    onsets_by_time = defaultdict(set)
    for note in note_events:
        onsets_by_time[note["rounded_start"]].add(note["pitch"]) # match the rounded start time to the pitch name


    s_notes = []
    s_isOnset = []

    for t in j_offsets:
        t = round(t, 6)
        notes_sounding = [n["pitch"] for n in note_events if n["start"] <= t < n["end"]] #collect the sounding notes at time t
        notes_sounding = sorted(set(notes_sounding))
        onsets_now = onsets_by_time.get(t, set()) # get the onsets at time t
        
        s_notes.append(notes_sounding)
        s_isOnset.append([pitch in onsets_now for pitch in notes_sounding])

    return s_notes, s_isOnset

def get_durations(df, step_size=0.125):
    """
    Calculates the duration of each chord at each time step.
    args:
        df (pd.DataFrame): the DataFrame being constructed for the MIDI file.
        step_size (float): Time step size in seconds. Default is 0.125 (1/8 note).
    returns:
        s_durations (list): List of durations for each chord at each time step.
    """

    s_durations = []
    current_notes = None
    count = 0

    for _, row in df.iterrows():
        notes = tuple(sorted(row["s_notes"]))

        if notes != current_notes: # if the notes have changed
            if count > 0: 
                duration = round(count * step_size, 6) #calculate the duration for the previous group of notes
                s_durations.extend([duration] * count) 
            current_notes = notes # update the current notes to the new ones
            count = 1
        else:
            count += 1 # if the notes are the same, increase the count

    if count > 0: #handle the last group of notes
        duration = round(count * step_size, 6)
        s_durations.extend([duration] * count)

    return s_durations

def get_measures(df, midi_stream, step_size=0.125):
    """
    Calcualtes the measure number for each time step.
    args:
        df (pd.DataFrame): the DataFrame being constructed for the MIDI file.
        midi_stream (PrettyMIDI): PrettyMIDI object representing the MIDI file.
        step_size (float): Time step size in seconds. Default is 0.125 (1/8 note).
    returns:
        list: A list of measure numbers for each time step in the Dataframe
    """

    steps_per_beat = int(1 / step_size)        
    beats_per_measure = 4                       # assuming 4/4
    time_signatures = midi_stream.time_signature_changes, #time signature changes in the MIDI file 
    if len(midi_stream.time_signature_changes) > 0:
        time_signature = time_signatures[0][0].numerator
        beats_per_measure = time_signature
    
    steps_per_measure = beats_per_measure * steps_per_beat  # 4 * 8

    return [1 + (i // steps_per_measure) for i in range(len(df))]

def extract_midi_info(file_path, step_size=0.125):
    """
    Extracts MIDI information from a specific .mid file.
    args: 
        file_path (str): Path to the MIDI file.
        step_size (float): Time step size in seconds for the analysis (default: 0.125, 1/8 note)
    returns: 
        pd.DataFrame: A DataFrame containing the extracted midi info.
    """

    df = pd.DataFrame() #initialize and empty DataFrame
    midi_stream = pm.PrettyMIDI(file_path) #load the MIDI file with pretty_midi

    df['j_offset'] = get_offsets(midi_stream)

    seen = set() # track unique note events
    note_events = []
    for instrument in midi_stream.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch = pm.note_number_to_name(note.pitch)
                start = round(round(note.start / step_size) * step_size, 6)
                end = round(note.end, 6)
                key = (pitch, start)
                if key not in seen:
                    seen.add(key)
                    note_events.append({"pitch": pitch, "start": start, "end": end})

    note_events.sort(key=lambda x: x["start"]) #sort notes by their start time

    df['s_notes'], df["s_isOnset"] = get_notes_onsets(note_events=note_events, j_offsets=df["j_offset"])

    df["s_duration"] = get_durations(df)

    df["s_measure"] = get_measures(df=df, midi_stream=midi_stream)

    return df

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