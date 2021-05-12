import tensorflow as tf
from scipy.io.wavfile import read, write
from sphfile import SPHFile

sph = SPHFile('z9z6531a.wav')
sph.write_wav( 'test.wav')
rate, data = read('test.wav')

print(tf.__version__)
print(read)