import librosa as la
import librosa.display
import numpy as np
import scipy
import wave
import matplotlib.pyplot as plt

y, sr = la.core.load(
    'exp/rnn_irm/decode_ans_C1/single_mix_test/raw_audio_0.wav',sr=16000)
y2, sr2 = la.core.load(
    '/mnt/d/tf_recipe/IRM_Speech_Enhancement/_decode_index/speech1_16k.wav',sr=16000)
print(len(y))
y1 = y[int(len(y)*0.41):int(len(y)*0.62)]
while len(y1) < len(y2):
  np.tile(y1, 2)
y1 = y1[:len(y2)]
y = (y1*1.4+y2)/2.4
y = y/np.max(np.abs(y)) * 32767

# librosa.output.write_wav(
#     '/mnt/d/tf_recipe/IRM_Speech_Enhancement/_decode_index/speech3.wav', y, sr=sr)
wavefile = wave.open(
    '/mnt/d/tf_recipe/IRM_Speech_Enhancement/_decode_index/speech3.wav', 'wb')
nframes = len(y)
wavefile.setparams((1, 2, 16000, nframes,
                    'NONE', 'not compressed'))
wavefile.writeframes(
    np.array(y, dtype=np.int16))

plt.figure(figsize=(20, 5))
spec = np.sqrt(np.sqrt(np.abs(
    la.stft(y, n_fft=512, hop_length=256, window=scipy.signal.windows.hann))))
la.display.specshow(spec)
plt.show()
plt.close()


# y,sr=la.core.load('exp/rnn_irm/decode_ans_C1/single_mix_test/restore_audio_2.wav')
# # y=y[50000:140000]
plt.figure(figsize=(20, 5))
la.display.specshow(np.sqrt(np.sqrt(np.abs(
    la.stft(y2, n_fft=512, hop_length=256, window=scipy.signal.windows.hann)))))
plt.show()
plt.close()

# y,sr=la.core.load('exp/rnn_irm/decode_ans_C1/single_mix_test/restore_audio_0_other.wav')
# y=y[50000:440000]
# plt.figure(figsize=(20, 5))
# la.display.specshow(np.sqrt(np.sqrt(np.abs(la.stft(y,n_fft=512,hop_length=256,window=scipy.signal.windows.hann)))))
# plt.show()
# plt.close()

# Visualize an STFT power spectrum

'''
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
y, sr = librosa.load('exp/rnn_irm/decode_ans_C1/single_mix_test/raw_audio_0.wav')
plt.figure(figsize=(12, 8))

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')

# Or on a logarithmic scale

plt.subplot(4, 2, 2)
librosa.display.specshow(D, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')

# Or use a CQT scale

CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
plt.subplot(4, 2, 3)
librosa.display.specshow(CQT, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (note)')

plt.subplot(4, 2, 4)
librosa.display.specshow(CQT, y_axis='cqt_hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (Hz)')

# Draw a chromagram with pitch classes

C = librosa.feature.chroma_cqt(y=y, sr=sr)
plt.subplot(4, 2, 5)
librosa.display.specshow(C, y_axis='chroma')
plt.colorbar()
plt.title('Chromagram')

# Force a grayscale colormap (white -> black)

plt.subplot(4, 2, 6)
librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear power spectrogram (grayscale)')

# Draw time markers automatically

plt.subplot(4, 2, 7)
librosa.display.specshow(D, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log power spectrogram')

# Draw a tempogram with BPM markers

plt.subplot(4, 2, 8)
Tgram = librosa.feature.tempogram(y=y, sr=sr)
librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
plt.colorbar()
plt.title('Tempogram')
plt.tight_layout()

# Draw beat-synchronous chroma in natural time

plt.figure()
tempo, beat_f = librosa.beat.beat_track(y=y, sr=sr, trim=False)
beat_f = librosa.util.fix_frames(beat_f, x_max=C.shape[1])
Csync = librosa.util.sync(C, beat_f, aggregate=np.median)
beat_t = librosa.frames_to_time(beat_f, sr=sr)
ax1 = plt.subplot(2,1,1)
librosa.display.specshow(C, y_axis='chroma', x_axis='time')
plt.title('Chroma (linear time)')
ax2 = plt.subplot(2,1,2, sharex=ax1)
librosa.display.specshow(Csync, y_axis='chroma', x_axis='time',
                         x_coords=beat_t)
plt.title('Chroma (beat time)')
plt.tight_layout()
'''
