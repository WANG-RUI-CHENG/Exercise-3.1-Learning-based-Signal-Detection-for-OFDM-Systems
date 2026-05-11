from __future__ import division
import numpy as np
# import scipy.interpolate
# import tensorflow as tf
import math
import os

CR = 1


def print_something():
    print ('utils.py has been loaded perfectly')



def Clipping (x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL*sigma
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))
    return x_clipped


def PAPR(x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10*np.log10(PeakP/AvgP)
    return PAPR_dB


def Modulation(bits,mu):
    bit_r = bits.reshape((int(len(bits)/mu), mu))
    if mu == 2:
        # QPSK 沿用原始程式的 I/Q 映射，讓 task (b) 與原 reference 設定保持相容。
        return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)                                    # This is just for QAM modulation
    elif mu == 6:
        # 64-QAM 每個 symbol 需要 6 bits；前三個 bits 決定 I 軸，後三個 bits 決定 Q 軸。
        # 使用 Gray-coded 8-PAM 可讓相鄰星座點只差一個 bit，較符合常見 QAM mapping，BER 比較也較合理。
        # 星座點最後除以 sqrt(42) 做平均能量正規化，避免 64-QAM 因振幅較大而和 QPSK 產生不公平的 SNR 比較。
        # Mapping: 000,-7; 001,-5; 011,-3; 010,-1; 110,1; 111,3; 101,5; 100,7.
        level_map = {
            (0,0,0): -7, (0,0,1): -5, (0,1,1): -3, (0,1,0): -1,
            (1,1,0):  1, (1,1,1):  3, (1,0,1):  5, (1,0,0):  7,
        }
        i_part = np.array([level_map[tuple(b[:3].astype(int))] for b in bit_r])
        q_part = np.array([level_map[tuple(b[3:].astype(int))] for b in bit_r])
        return (i_part + 1j*q_part) / np.sqrt(42.0)
    else:
        raise ValueError('Only QPSK (mu=2) and 64-QAM (mu=6) are supported')


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time, CP, CP_flag, mu, K):

    if CP_flag == False:
        # add noise CP
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        codeword_noise = Modulation(bits_noise, mu)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]               # take the last CP samples ...
    #cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise


def removeCP(signal, CP, K):
    return signal[CP:(CP+K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)



def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


# 保留原始函式以維持程式結構；目前 DNN pipeline 直接輸出 bits，不會使用傳統 equalization 後取 payload 的流程。
def get_payload(equalized):
    return equalized[dataCarriers]



def PS(bits):
    return bits.reshape((-1,))


'''
def Modulation(bits, mu):


    bit_r = bits.reshape((int(len(bits)/mu), mu))
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)                                    # This is just for QAM modulation


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP+K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))
'''


def ofdm_simulate(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag):
    payloadBits_per_OFDM = mu*len(dataCarriers)
    # --- training inputs ----
    # DNN 的輸入由兩個接收 OFDM symbols 組成：一個含 pilot，用來讓網路隱含學習通道資訊；
    # 另一個含待偵測資料，用來恢復目標 bits。兩者各拆成 real/imag 後串接，因此輸入維度為 256。
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        QAM = Modulation(bits,mu)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue

    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K)
    #OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX,CR)                            # add clipping
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP,K)
    #OFDM_RX_noCP = removeCP(OFDM_RX)
    # ----- target inputs ---
    # 這個 OFDM symbol 承載真正要偵測的 codeword；label 則在 Train/Test 中由 pred_range 選出對應 bit 區段。
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword,mu)
    if len(codeword_qam) != K:
        # 若 codeword 長度與 K 不一致，代表 mu 或 payloadBits_per_OFDM 設定不一致；
        # 保留錯誤提示可快速檢查 modulation 與 OFDM subcarrier 數是否匹配。
        print ('length of code word is not equal to K, error !!')
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    #OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword,CR) # add clipping
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword,CP,K)
    #OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) #sparse_mask


'''

# Older alternatives from the original starter code are intentionally left commented out.

'''
