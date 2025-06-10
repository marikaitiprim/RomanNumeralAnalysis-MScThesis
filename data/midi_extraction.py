import os
import pretty_midi as pm

def extract_midi_info(file_path):
    """
    Extracts MIDI information from a specific .mid file.

    args: 
        file_path (str): Path to the MIDI file.

    returns: 
        midi_info (dict): Dictionary containing the MIDI file's information, including:
                            - total_duration (float): Total duration of the MIDI file in seconds.
                            - time_signature (list): Time signature changes in the MIDI file.
                            - instrument (list): List of instruments in the MIDI file that are not drums.
                            - key_signature (list): Key signature changes in the MIDI file.
                            - onsets (list): List of onset times for the MIDI file.
                            - notes (list): List of notes in the MIDI file, sorted by their start time.
    """
    midi_stream = pm.PrettyMIDI(file_path) #load the MIDI file with pretty_midi
    midi_info = {
        'total_duration': midi_stream.get_end_time(), #total duration of the MIDI file in seconds
        'time_signature': midi_stream.time_signature_changes, #time signature changes in the MIDI file - usually get midi_stream.time_signature_changes[0] 
        'instrument': [instrument for instrument in midi_stream.instruments if not instrument.is_drum], #store the instruments in the MIDI file that are not drums
        'key_signature': midi_stream.key_signature_changes, #key signature changes in the MIDI file
        'onsets': midi_stream.get_onsets(),
    }

    midi_info['notes'] = sorted([note for instrument in midi_info['instrument'] for note in instrument.notes], key=lambda x: x.start), # sort the notes by their start time

    return midi_info


if __name__ == "__main__":
    dataset = "/Users/marikaitiprimenta/Desktop/MSC-THESIS/ChordRecognition-MScThesis/dataset"
    for file in os.listdir(dataset):
        if file.endswith('.mid'):
            file_path = os.path.join(dataset, file)

            print("Processing file:", file)
    
            midi_info = extract_midi_info(file_path)
            print(f"Extracted MIDI info from {file}:")
            print(f"Total Duration: {midi_info['total_duration']} seconds")
            print(f"Time Signature Changes: {midi_info['time_signature']}")
            print(f"Number of Instruments: {len(midi_info['instrument'])}")
            print(f"Key Signature Changes: {midi_info['key_signature']}")
            print(f"Number of Onsets: {len(midi_info['onsets'])}")
            print(f"Number of Notes: {len(midi_info['notes'][0])}\n")




