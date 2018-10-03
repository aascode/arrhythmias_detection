import pywt._dwt as _dwt
import pywt._extensions._pywt as _pywt
import numpy as np


class Wavelet(object):
    """

    Examples
    --------
    >>> import pywt._dwt as _dwt
    >>> import pywt
    >>> import pywt._extensions._pywt as _pywt
    >>> wav = Wavelet(_input=ecg)
    >>> print(wav.twe, wav.rwe, wav.we)

    """
    def __init__(self, _input=None):
        self._input = _input
        if _input is None:
            self.l = None
            self.level = None
        else:
            self.l = len(_input)
            self.level = _dwt.dwt_max_level(data_len=self.l, filter_len=self.fl)
        self.fl = _pywt.Wavelet('haar').dec_len
        self.method = _pywt.MODES.smooth
        self.wavelet = 'haar'
        self.ca = []
        self.cd = []
        self.eng = []
        self.twe = []
        self.rwe = []
        self.we = []
        # self.wavelet_decompose()
        # self.total_wavelet_energy()
        # self.relative_wavelet_energy()
        # self.wavelet_entropy()
        pass

    def wavelet_decompose(self, _input, level):
        a = _input
        ca, cd = [], []
        for _ in range(level):
            a, d = _dwt.dwt(data=a, wavelet='bior6.8')
            ca = np.append(ca, a)
            cd = np.append(cd, d)
            self.eng.append(np.sum(np.square(d)))

        self.eng.append(np.sum(np.square(a)))
        return ca, cd, a, d

    def get_relative_wavelet_energy(self):
        return self.rwe

    def total_wavelet_energy(self):
        self.twe = np.sum(self.eng)

    def relative_wavelet_energy(self):
        self.rwe = np.divide(self.eng, self.twe)

    def wavelet_entropy(self):
        self.we = - np.sum(self.rwe * np.log(self.rwe))

    def get_total_wavelet_energy(self):
        return self.twe

    def get_wavelet_entropy(self):
        return self.we

    def get_normalized_we(self):
        return self.we / len(self.eng)
