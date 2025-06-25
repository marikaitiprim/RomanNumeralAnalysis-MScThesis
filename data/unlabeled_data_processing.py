import os
import pretty_midi as pm
import pandas as pd
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


def extract_midi_info(file_path, step_size=0.125, tempo=120):
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
    total_duration = midi_stream.get_end_time() #total duration of the MIDI file in seconds

    num_steps = int(total_duration // step_size) + 1
    j_offsets = [round(i * step_size, 6) for i in range(num_steps)] # divide the total duration into steps of step_size

    df['j_offset'] = j_offsets
    
    # time_signatures = midi_stream.time_signature_changes, #time signature changes in the MIDI file 
    # if len(midi_stream.time_signature_changes) > 0:
    #     time_signature = time_signatures[0]
    #     print("time signature:", time_signature)

    note_events = []
    for instrument in midi_stream.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note_events.append({"start": note.start, "end": note.end, "pitch": pm.note_number_to_name(note.pitch)})

    note_events.sort(key=lambda x: x["start"]) #sort notes by their start time

    s_notes = []
    for t in j_offsets:
        sounding_notes = [] #collect the sounding notes at time t
        for n in note_events:
            if n["start"] <= t < n["end"]: # detect if the note is sounding at time t
                sounding_notes.append(n["pitch"])
        s_notes.append(sorted(list(pd.unique(sounding_notes)))) # store unique sounding notes at time t

    df['s_notes'] = s_notes


    from collections import defaultdict

    # 1. Build fast lookup: time → set of onset pitches
    onsets_by_time = defaultdict(set)
    for n in note_events:
        rounded_start = round(round(n["start"] / step_size) * step_size, 6)  # same precision as j_offset
        onsets_by_time[rounded_start].add(n["pitch"])

    # 2. Fast loop over j_offset and s_notes
    s_isOnset = []
    for i, row in df.iterrows():
        t = round(row["j_offset"], 6)
        onset_flags = [pitch in onsets_by_time.get(t, set()) for pitch in row["s_notes"]]
        s_isOnset.append(onset_flags)

    df["s_isOnset"] = s_isOnset

    

    s_durations = []
    count = 0

    for i, row in df.iterrows():
        onset = any(row["s_isOnset"])

        if onset and count > 0:
            duration = round(count * step_size, 6)
            s_durations.extend([duration] * count)
            count = 1
        elif onset:
            count = 1
        elif len(row["s_notes"]) == 0:
            # Silence: flush group, set 0 duration
            if count > 0:
                duration = round(count * step_size, 6)
                s_durations.extend([duration] * count)
                count = 0
            s_durations.append(0.0)
        else:
            count += 1

    # Final group
    if count > 0:
        duration = round(count * step_size, 6)
        s_durations.extend([duration] * count)

    # Safety padding
    while len(s_durations) < len(df):
        s_durations.append(0.0)

    df["s_duration"] = s_durations

    # s_durations = []
    # count = 0
    # for i, row in df.iterrows():
    #     is_onset = any(row["s_isOnset"])  # if any note is starting

    #     if is_onset and count > 0:
    #         s_durations.extend([count * step_size] * count)
    #         count = 1
    #     elif is_onset:
    #         count = 1
    #     else:
    #         count += 1

    # # Final group
    # if count > 0:
    #     s_durations.extend([count * step_size] * count)

    # df["s_duration"] = s_durations

    # s_durations = []
    # current_notes = None
    # count = 0

    # for _, row in df.iterrows():
    #     notes = tuple(sorted(row["s_notes"])) if row["s_notes"] else None

    #     if notes is None: # no notes sounding at this time step
    #         if count > 0:
    #             duration = round(count * step_size, 6)
    #             s_durations.extend([duration] * count)
    #             count = 0

    #         s_durations.append(0.0)
    #         current_notes = None
    #         continue

    #     if notes != current_notes: # new notes sounding at this time step
    #         if count > 0:
    #             duration = round(count * step_size, 6)
    #             s_durations.extend([duration] * count)
    #         current_notes = notes
    #         count = 1
    #     else: # same notes as before
    #         count += 1

    # if count > 0: #finalize the last group of notes
    #     duration = round(count * step_size, 6)
    #     s_durations.extend([duration] * count)

    # while len(s_durations) < len(df): #ensure all time steps have a duration 
    #     s_durations.append(0.0)

    # df["s_duration"] = s_durations

    #     row = {
    #         's_measure': measure,
    #         's_isOnset': is_onset,
    #         'a_localKey': local_key
    
    #     }
    #     rows.append(row)

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