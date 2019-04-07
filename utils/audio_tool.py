import soundfile as sf
from mir_eval.separation import bss_eval_sources
import numpy as np
import FLAGS
import librosa
AMP_MAX = (2 ** (FLAGS.MIXED_AISHELL_PARAM.AUDIO_BITS - 1) - 1)

'''
soundfile.info(file, verbose=False)
soundfile.available_formats()
soundfile.available_subtypes(format=None)
soundfile.read(file, frames=-1, start=0, stop=None, dtype='float64', always_2d=False, fill_value=None, out=None, samplerate=None, channels=None, format=None, subtype=None, endian=None, closefd=True)
soundfile.write(file, data, samplerate, subtype=None, endian=None, format=None, closefd=True)
'''


def read_audio(file):
  data, sr = sf.read(file)
  if sr != FLAGS.MIXED_AISHELL_PARAM.FS:
    data = librosa.resample(data, sr, FLAGS.MIXED_AISHELL_PARAM.FS, res_type='kaiser_fast')
    print('resample wav(%d to %d) :' % (sr, FLAGS.MIXED_AISHELL_PARAM.FS), file)
    # librosa.output.write_wav(file, data, FLAGS.PARAM.FS)
  return data*AMP_MAX, FLAGS.PARAM.FS


def write_audio(file, data, sr):
  return sf.write(file, data/AMP_MAX, sr)


def cal_SDR(src_ref, src_deg):
    """Calculate Source-to-Distortion Ratio(SDR).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_deg: numpy.ndarray, [C, T], reordered by best PIT permutation
    Returns:
        SDR
    """
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_deg)
    return sdr
