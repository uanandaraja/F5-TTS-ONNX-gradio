import gradio as gr
import json
import os
import re
import sys
import time
import jieba
import numpy as np
import onnxruntime
import torch
import torchaudio
from pypinyin import lazy_pinyin, Style

config_file = "config.json"


def load_config():
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            return json.load(file)
    return {}


def save_config(gentxt, vocpath, oma, omb, omc, refa, gena, reftxt):
    config = {
        "gentxt": gentxt,
        "vocpath": vocpath,
        "oma": oma,
        "omb": omb,
        "omc": omc,
        "refa": refa,
        "gena": gena,
        "reftxt": reftxt,
    }
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)
    return config


def is_chinese_char(c):
    cp = ord(c)
    return (
        0x4E00 <= cp <= 0x9FFF or
        0x3400 <= cp <= 0x4DBF or
        0x20000 <= cp <= 0x2A6DF or
        0x2A700 <= cp <= 0x2B73F or
        0x2B740 <= cp <= 0x2B81F or
        0x2B820 <= cp <= 0x2CEAF or
        0xF900 <= cp <= 0xFAFF or
        0x2F800 <= cp <= 0x2FA1F
    )


def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    merged_trans = str.maketrans({
        '"': '"', '"': '"', ''': "'", ''': "'",
        ';': ','
    })
    chinese_punctuations = set("。，、；：？！《》【】—…")
    for text in text_list:
        char_list = []
        text = text.translate(merged_trans)
        for seg in jieba.cut(text):
            if seg.isascii():
                if char_list and len(seg) > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and all(is_chinese_char(c) for c in seg):
                pinyin_list = lazy_pinyin(
                    seg, style=Style.TONE3, tone_sandhi=True)
                for c in pinyin_list:
                    if c not in chinese_punctuations:
                        char_list.append(" ")
                    char_list.append(c)
            else:
                for c in seg:
                    if c.isascii():
                        char_list.append(c)
                    elif c in chinese_punctuations:
                        char_list.append(c)
                    else:
                        char_list.append(" ")
                        pinyin = lazy_pinyin(
                            c, style=Style.TONE3, tone_sandhi=True)
                        char_list.extend(pinyin)
        final_text_list.append(char_list)
    return final_text_list


def list_str_to_idx(text, vocab_char_map, padding_value=-1):
    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor(
        [get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    text = torch.nn.utils.rnn.pad_sequence(
        list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


def process_audio(gentxt, vocpath, oma, omb, omc, refa, gena, reftxt):
    save_config(gentxt, vocpath, oma, omb, omc, refa, gena, reftxt)

    HOP_LENGTH = 256
    SAMPLE_RATE = 24000
    RANDOM_SEED = 9527
    NFE_STEP = 32
    dynamic_axes = False
    SPEED = 1.0

    with open(vocpath, "r", encoding="utf-8") as f:
        vocab_char_map = {char[:-1]: i for i, char in enumerate(f)}
    vocab_size = len(vocab_char_map)

    # ONNX Runtime settings
    onnxruntime.set_seed(RANDOM_SEED)
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 3
    session_opts.inter_op_num_threads = 0
    session_opts.intra_op_num_threads = 0
    session_opts.enable_cpu_mem_arena = True
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.add_session_config_entry(
        "session.intra_op.allow_spinning", "1")
    session_opts.add_session_config_entry(
        "session.inter_op.allow_spinning", "1")

    # Initialize sessions
    ort_session_A = onnxruntime.InferenceSession(
        oma, sess_options=session_opts, providers=['CPUExecutionProvider'])
    ort_session_B = onnxruntime.InferenceSession(
        omb, sess_options=session_opts, providers=['CPUExecutionProvider'])
    ort_session_C = onnxruntime.InferenceSession(
        omc, sess_options=session_opts, providers=['CPUExecutionProvider'])

    # Get input/output names
    in_name_A = ort_session_A.get_inputs()
    out_name_A = ort_session_A.get_outputs()
    in_name_B = ort_session_B.get_inputs()
    out_name_B = ort_session_B.get_outputs()
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()

    # Process audio
    audio, sr = torchaudio.load(refa)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=SAMPLE_RATE)
        audio = resampler(audio)
    audio = audio.unsqueeze(0).numpy()

    model_type = ort_session_A._inputs_meta[0].type
    if "float16" in model_type:
        audio = audio.astype(np.float16)

    zh_pause_punc = r"。，、；：？！"
    ref_text_len = len(reftxt.encode('utf-8')) + 3 * \
        len(re.findall(zh_pause_punc, reftxt))
    gen_text_len = len(gentxt.encode('utf-8')) + 3 * \
        len(re.findall(zh_pause_punc, gentxt))
    ref_audio_len = audio.shape[-1] // HOP_LENGTH + 1
    max_duration = np.array(ref_audio_len + int(ref_audio_len /
                            ref_text_len * gen_text_len / SPEED), dtype=np.int64)

    text = convert_char_to_pinyin([reftxt + gentxt])
    text_ids = list_str_to_idx(text, vocab_char_map).numpy()
    time_step = np.array(0, dtype=np.int32)

    # Run inference
    start_count = time.time()
    noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, qk_rotated_empty, ref_signal_len = ort_session_A.run(
        [out.name for out in out_name_A],
        {
            in_name_A[0].name: audio,
            in_name_A[1].name: text_ids,
            in_name_A[2].name: max_duration
        })

    while time_step < NFE_STEP:
        noise = ort_session_B.run(
            [out_name_B[0].name],
            {name.name: val for name, val in zip(in_name_B,
                                                 [noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, qk_rotated_empty, time_step])})[0]
        time_step += 1

    generated_signal = ort_session_C.run(
        [out_name_C[0].name],
        {in_name_C[0].name: noise, in_name_C[1].name: ref_signal_len})[0]

    end_count = time.time()

    # Save audio
    audio_tensor = torch.tensor(
        generated_signal, dtype=torch.float32).squeeze(0)
    torchaudio.save(gena, audio_tensor, SAMPLE_RATE)

    return f"Audio generation complete. Time taken: {end_count - start_count:.3f} seconds"


config = load_config()

interface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Textbox(value=config.get(
            "gentxt", "write what you want generated"), label="Generation Text"),
        gr.Textbox(value=config.get(
            "vocpath", "./models/vocab.txt"), label="Vocab Path"),
        gr.Textbox(value=config.get(
            "oma", "./models/onnx/F5_Preprocess.onnx"), label="Model A Path"),
        gr.Textbox(value=config.get(
            "omb", "./models/onnx/F5_Transformer.onnx"), label="Model B Path"),
        gr.Textbox(value=config.get(
            "omc", "./models/onnx/F5_Decode.onnx"), label="Model C Path"),
        gr.Textbox(value=config.get("refa", "./audio/sample.wav"),
                   label="Reference Audio Path"),
        gr.Textbox(value=config.get(
            "gena", "./audio/generated/generated_audio.wav"), label="Generated Audio Path"),
        gr.Textbox(value=config.get(
            "reftxt", "And now, coming to you from the classiest station on the air, this is "), label="Reference Text"),
    ],
    outputs="text",
    title="F5-TTS-ONNX GUI",
    description="Text-to-Speech Generation Interface"
)

if __name__ == "__main__":
    interface.launch()
