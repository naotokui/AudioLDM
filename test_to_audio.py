#!/usr/bin/python3
#%%
import os
from audioldm import text_to_audio, build_model, save_wave

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-t",
    "--text",
    type=str,
    required=False,
    default="A hammer is hitting a wooden surface",
    help="Text prompt to the model for audio generation",
)

parser.add_argument(
    "-s",
    "--save_path",
    type=str,
    required=False,
    help="The path to save model output",
    default="./output",
)

parser.add_argument(
    "-ckpt",
    "--ckpt_path",
    type=str,
    required=False,
    help="The path to the pretrained .ckpt model",
    default=os.path.join(
                os.path.expanduser("~"),
                ".cache/audioldm/audioldm-s-full.ckpt",
            ),
)

parser.add_argument(
    "-b",
    "--batchsize",
    type=int,
    required=False,
    default=1,
    help="Generate how many samples at the same time",
)

parser.add_argument(
    "-gs",
    "--guidance_scale",
    type=float,
    required=False,
    default=2.5,
    help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
)

parser.add_argument(
    "-dur",
    "--duration",
    type=float,
    required=False,
    default=10.0,
    help="The duration of the samples",
)

parser.add_argument(
    "-n",
    "--n_candidate_gen_per_text",
    type=int,
    required=False,
    default=3,
    help="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
)

parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default=42,
    help="Change this value (any integer number) will lead to a different generation result.",
)

args = parser.parse_args()

assert args.duration % 2.5 == 0, "Duration must be a multiple of 2.5"

#%%
import librosa
import pyrubberband as pyrb
import madmom
from madmom.features.downbeats import RNNDownBeatProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
import soundfile as sf

proc = DBNDownBeatTrackingProcessor(beats_per_bar=4, fps = 100)

def one_bar_segment(output_dir, filename):
    file_path = os.path.join(output_dir, filename)
    try:
        y, sr = librosa.core.load(file_path, sr=None) # sr = None will retrieve the original sampling rate = 44100
    except:
        print('load file failed')
        return
    try:
        act = RNNDownBeatProcessor()(file_path)
        down_beat=proc(act) # [..., 2] 2d-shape numpy array
    except:
        print('except happended')
        return
    #print(down_beat)
    #print(len(y) / sr)
    #import pdb; pdb.set_trace()
    #retrieve 1, 2, 3, 4, 1blocks
    count = 0
    #print(file)
    name = filename.replace('.wav', '')
    print(down_beat)
    for i in range(down_beat.shape[0]):
        if down_beat[i][1] == 1 and i + 4 < down_beat.shape[0] and down_beat[i+4][1] == 1:
            print(down_beat[i: i + 5, :])
            start_time = down_beat[i][0]
            end_time = down_beat[i + 4][0]
            count += 1
            out_path = os.path.join(output_dir, f'{name}_{count}.wav')
            #print(len(y) / sr)
            #print(sr)
            y_one_bar, _ = librosa.core.load(file_path, offset=start_time, duration = end_time - start_time, sr=None)
            y_stretch = pyrb.time_stretch(y_one_bar, sr,  (end_time - start_time) / 2)
            #print((end_time - start_time))
            #print()
            sf.write(out_path, y_stretch, sr)
            print('save file: ',  f'{name}_{count}.wav')
            #y, sr = librosa.core.load(out_path, sr=None)
            #print(librosa.get_duration(y, sr=sr))
            return out_path

from glob import glob

def extract_loops(output_dir, filename):
    filepaths = glob(os.path.join(output_dir, "%s_*.wav" % filename))
    outputs = []
    for filepath in filepaths:
        path = one_bar_segment(output_dir, os.path.basename(filepath))
        outputs.append(path)
    return outputs

#%%

text = "an amen break drum loop"

save_path = "./output"
random_seed = 42
duration = 5.0
guidance_scale = 2.5
n_candidate_gen_per_text = 4
ckpt_path = os.path.join(
                os.path.expanduser("~"),
                ".cache/audioldm/audioldm-s-full.ckpt",
            )
os.makedirs(save_path, exist_ok=True)
audioldm = build_model(ckpt_path=ckpt_path)
batch_size = 4

waveform = text_to_audio(
    audioldm,
    text,
    random_seed,
    duration=duration,
    guidance_scale=guidance_scale,
    n_candidate_gen_per_text=n_candidate_gen_per_text,
    batchsize=batch_size,
)

save_wave(waveform, save_path, name=text)

# %%
print(audioldm)

# %%


def generate_loop(unused_addr, text, batch_size=1, deck="A"):
    waveform = text_to_audio(
        audioldm,
        text,
        random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=n_candidate_gen_per_text,
        batchsize=batch_size,
    )
    save_wave(waveform, save_path, name=text)
    filepaths = extract_loops("./output/", text)

    if len(filepaths) > 0:
        client.send_message("/generated", (deck, filepaths[0]))


# %%

from pythonosc import dispatcher
from pythonosc import osc_server, udp_client

import time
from threading import Thread

client = udp_client.SimpleUDPClient('127.0.0.1', 10018)
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/generate_loop", generate_loop)

def beacon():
    while True:
        client.send_message("/heartbeat", 1)
        time.sleep(1.0)
t1 = Thread(target = beacon)
t1.setDaemon(True)
t1.start()

server = osc_server.ThreadingOSCUDPServer(
    ('localhost', 10017), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()
# %%
