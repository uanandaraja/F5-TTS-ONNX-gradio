import gradio as gr
import json
import os
import time
import numpy as np
import onnxruntime
import torch
import torchaudio
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

config_file = "config.json"

def load_config():
    if os.path.exists(config_file):
        logger.info("Loading configuration from file")
        with open(config_file, "r") as file:
            return json.load(file)
    logger.warning("No configuration file found, using defaults")
    return {}

def save_config(gentxt, vocpath, oma, omb, omc, refa, gena, reftxt):
    logger.info("Saving configuration to file")
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

def list_str_to_idx(text, vocab_char_map, padding_value=-1):
    logger.debug(f"Converting text to indices: {text}")
    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor([get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    text = torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

def process_audio(gentxt, vocpath, oma, omb, omc, refa, gena, reftxt, progress=gr.Progress()):
    try:
        logger.info("Starting audio processing")
        logger.info(f"Generation text: {gentxt}")
        logger.info(f"Reference text: {reftxt}")

        save_config(gentxt, vocpath, oma, omb, omc, refa, gena, reftxt)

        HOP_LENGTH = 256
        SAMPLE_RATE = 24000
        RANDOM_SEED = 9527
        NFE_STEP = 32
        SPEED = 1.0

        # Loading configuration and models with progress tracking
        progress(0, desc="Initialization")
        logger.info("Loading vocabulary")
        with open(vocpath, "r", encoding="utf-8") as f:
            vocab_char_map = {char[:-1]: i for i, char in enumerate(f)}
        logger.info(f"Vocabulary size: {len(vocab_char_map)}")
        progress(0.25)

        # ONNX Runtime settings
        logger.info("Configuring ONNX Runtime")
        onnxruntime.set_seed(RANDOM_SEED)
        session_opts = onnxruntime.SessionOptions()
        session_opts.log_severity_level = 3
        session_opts.inter_op_num_threads = 0
        session_opts.intra_op_num_threads = 0
        session_opts.enable_cpu_mem_arena = True
        session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
        session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
        progress(0.5)

        # Initialize sessions
        logger.info("Loading ONNX models")
        ort_session_A = onnxruntime.InferenceSession(oma, sess_options=session_opts, providers=['CPUExecutionProvider'])
        ort_session_B = onnxruntime.InferenceSession(omb, sess_options=session_opts, providers=['CPUExecutionProvider'])
        ort_session_C = onnxruntime.InferenceSession(omc, sess_options=session_opts, providers=['CPUExecutionProvider'])
        progress(0.75)

        # Get input/output names
        in_name_A = ort_session_A.get_inputs()
        out_name_A = ort_session_A.get_outputs()
        in_name_B = ort_session_B.get_inputs()
        out_name_B = ort_session_B.get_outputs()
        in_name_C = ort_session_C.get_inputs()
        out_name_C = ort_session_C.get_outputs()
        progress(1.0)

        # Audio processing
        progress(0, desc="Audio Processing")
        logger.info(f"Loading reference audio from {refa}")
        audio, sr = torchaudio.load(refa)
        if sr != SAMPLE_RATE:
            logger.info(f"Resampling audio from {sr}Hz to {SAMPLE_RATE}Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio = resampler(audio)
        audio = audio.unsqueeze(0).numpy()
        progress(0.33)

        # Check model type safely
        model_meta = ort_session_A.get_inputs()[0].type
        if model_meta and "float16" in model_meta:
            logger.info("Converting audio to float16")
            audio = audio.astype(np.float16)

        ref_text_len = len(reftxt)
        gen_text_len = len(gentxt)
        ref_audio_len = audio.shape[-1] // HOP_LENGTH + 1
        max_duration = np.array(ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED), dtype=np.int64)
        logger.info(f"Calculated max duration: {max_duration}")
        progress(0.66)

        # Prepare text input
        logger.info("Preparing text input")
        text = [list(reftxt + gentxt)]
        text_ids = list_str_to_idx(text, vocab_char_map).numpy()
        time_step = np.array(0, dtype=np.int32)
        progress(1.0)

        # Run inference
        logger.info("Starting inference")
        start_count = time.time()

        progress(0, desc="Model A (Preprocessing)")
        logger.info("Running Model A (Preprocessing)")
        noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, qk_rotated_empty, ref_signal_len = ort_session_A.run(
            [out.name for out in out_name_A],
            {
                in_name_A[0].name: audio,
                in_name_A[1].name: text_ids,
                in_name_A[2].name: max_duration
            })
        progress(1.0)

        logger.info(f"Running Model B (Transformer) for {NFE_STEP} steps")
        for step in progress.tqdm(range(NFE_STEP), desc="Model B (Transformer)"):
            noise = ort_session_B.run(
                [out_name_B[0].name],
                {name.name: val for name, val in zip(in_name_B, [noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, qk_rotated_empty, time_step])})[0]
            time_step += 1

        progress(0, desc="Model C (Decoding)")
        logger.info("Running Model C (Decoding)")
        generated_signal = ort_session_C.run(
            [out_name_C[0].name],
            {in_name_C[0].name: noise, in_name_C[1].name: ref_signal_len})[0]
        progress(1.0)

        end_count = time.time()
        inference_time = end_count - start_count
        logger.info(f"Inference completed in {inference_time:.3f} seconds")

        # Save audio
        progress(0, desc="Saving Audio")
        logger.info(f"Saving generated audio to {gena}")
        audio_tensor = torch.tensor(generated_signal, dtype=torch.float32).squeeze(0)
        torchaudio.save(gena, audio_tensor, SAMPLE_RATE)
        progress(1.0)

        logger.info("Audio generation process completed successfully")
        return f"Audio generation complete. Time taken: {inference_time:.3f} seconds"

    except Exception as e:
        logger.error(f"Error during audio processing: {str(e)}", exc_info=True)
        raise

config = load_config()

interface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Textbox(value=config.get("gentxt", "write what you want generated"), label="Generation Text"),
        gr.Textbox(value=config.get("vocpath", "./models/vocab.txt"), label="Vocab Path"),
        gr.Textbox(value=config.get("oma", "./models/onnx/F5_Preprocess.onnx"), label="Model A Path"),
        gr.Textbox(value=config.get("omb", "./models/onnx/F5_Transformer.onnx"), label="Model B Path"),
        gr.Textbox(value=config.get("omc", "./models/onnx/F5_Decode.onnx"), label="Model C Path"),
        gr.Textbox(value=config.get("refa", "./audio/sample.wav"), label="Reference Audio Path"),
        gr.Textbox(value=config.get("gena", "./audio/generated/generated_audio.wav"), label="Generated Audio Path"),
        gr.Textbox(value=config.get("reftxt", "And now, coming to you from the classiest station on the air, this is "), label="Reference Text"),
    ],
    outputs="text",
    title="F5-TTS-ONNX GUI",
    description="Text-to-Speech Generation Interface"
)

if __name__ == "__main__":
    logger.info("Starting F5-TTS-ONNX GUI")
    interface.launch()
