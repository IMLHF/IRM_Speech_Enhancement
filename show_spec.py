import librosa as la
import librosa.display
import numpy as np
import scipy
import wave
import matplotlib.pyplot as plt
import utils
import utils.audio_tool
import utils.spectrum_tool

y, sr = la.core.load(
    'exp/rnn_irm/target/C7_fu0.3_iter7.wav', sr=16000)
# utils.audio_tool.write_audio("_decode_index/speech0_16k.wav",y[:16000*10],16000,16,'wav',norm=False)
# exit(0)


# print(len(y))
# y1 = y[int(len(y)*0.41):int(len(y)*0.62)] # noise
# while len(y1) < len(y2):
#   np.tile(y1, 2)
# y1 = y1[:len(y2)]
# y = (y1*1.4+y2)/2.4
# y = y/np.max(np.abs(y)) * 32767


plt.figure(figsize=(12, 5))
# y=y[20000:115000]
y*=32767
print(len(y))
spec = utils.spectrum_tool.magnitude_spectrum_librosa_stft(y,512,256).T
# print('spec_max:',np.max(spec)) # 532707.0
spec_show=spec
# # spec_show = spec[:,150:950]
# # la.display.specshow(spec_show)
# # plt.show()
# # plt.close()
# la.display.specshow(np.log10(0.5+spec_show))
# la.display.specshow(np.sqrt(np.sqrt(spec)))
utils.spectrum_tool.picture_spec(np.log10(0.5+spec_show.T),'C7_fu0.3_iter7_')
# plt.savefig('exp/rnn_irm/target/C7_fu0.3_iter7_.jpg')
# # plt.show()
plt.close()

# # 减弱第一第二谐波
# con0 = np.array([0]*237,dtype=np.int)
# con0 = np.tile(con0,(np.shape(spec)[1],1)).T
# print(np.shape(con0))
# con1 = np.array([1]*20,dtype=np.int)
# con1 = np.tile(con1,(np.shape(spec)[1],1)).T
# con=np.concatenate([con1,con0],axis=0)
# print(np.shape(con))
# spec = np.where(con,spec/200,spec)
# la.display.specshow(np.log10(0.5+spec[:,150:950]))
# plt.show()
# plt.close()

# spec_complex = utils.spectrum_tool.griffin_lim(spec.T,512,256,50)
# # spec_complex=spec
# y_r = utils.spectrum_tool.librosa_istft(spec_complex,512,256)
# print(np.max(y),np.min(y))
# print(np.max(y_r),np.min(y_r))
# utils.audio_tool.write_audio('test.wav',y,16000,16,'wav')
# utils.audio_tool.write_audio('test_r.wav',y_r,16000,16,'wav')
