import wave
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # wav파일 읽기
  wav_file = '../data/wav/BASIC5000_0001.wav'

  # 분석 시각 .BASIC5000_0001.wav에서는 아래 시각에서 음소 "o"을 발화한다.
  target_time = 0.58

  # FFT(고속 푸리에 변환) 한 범위의 샘플 수
  # 2의 제곱이여야 한다.
  fft_size = 1024

  # 시각화 결과 파일(png파일)
  out_plot = './spectrum.png'

  # wav 파일을 열고, 아래 코드들을 수행한다.
  with wave.open(wav_file) as wav:
    # 샘플링 주파수[Hz] 확인
    sampling_frequency = wav.getframerate()
    # wav 데이터 읽기
    waveform = wav.readframes(wav.getnframes())
    # 읽어온 데이터는 바이너리 값이므로 정수로 변환한다.
    waveform = np.frombuffer(waveform, dtype=np.int16)

  # 분석 시간을 샘플 번호로 변환
  target_index = np.int(target_time * sampling_frequency)
  # FFT를 실행하는 구간만큼의 파형 데이터를 도출한다.
  frame = waveform[target_index: target_index + fft_size]
  # FFT 적용
  spectrum = np.fft.fft(frame)
  # 진폭 스펙트럼 확인
  absolute = np.abs(spectrum)
  # 진폭 스펙트럼은 좌우 대칭이므로 좌측 반만 이용한다.
  absolute = absolute[:np.int(fft_size/2) + 1]
  # 로그 함수를  취하고, 로그 진폭 스펙트럼 계산
  log_absolute = np.log(absolute + 1E-7)

  # 시간 파형과 로그 진폭 스펙트럼을 시각화한다.
  # 시각화 영역 생성
  plt.figure(figsize=(10, 10))
  # 그림 영역을 종으로 2분할하여 위쪽에 시간 파형을 그린다.
  plt.subplot(2, 1, 1)

  # 2분할한 그림 영역 밑에 로그 진폭 스펙트럼을 그린다.
  plt.subplot(2, 1, 2)
  # 횡축(주파수 축) 생성
  freq_axis = np.arange(np.int(fft_size/2)+1) \
              * sampling_frequency / fft_size
  # 로그 진폭 스펙트럼 시각화
  plt.plot(freq_axis, log_absolute)
  # 시각화한 그림의 제목과 횡축, 종축 라벨 정의
  plt.title('log-absolute spectrum')
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('Value')
  # 횡축 표시 영역을 0 ~최대 주파수로 제한
  plt.xlim([0, sampling_frequency/2])

  # 시각화된 결과물 저장
  plt.savefig(out_plot)