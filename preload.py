from TTS.utils.manage import ModelManager

# Downloads and caches the model ahead of time
model_manager = ModelManager()
model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")