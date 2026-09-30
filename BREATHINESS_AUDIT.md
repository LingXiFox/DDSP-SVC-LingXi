# Breathiness / Voice Quality Audit

## Scope

- Fixed matched validation set: 3 clips x A/B/C, mono 44.1 kHz.
- A = Realism OFF; B = Public Prior; C = Personalized best@405.
- All three paths use the same robotized controls, units, clip bounds and sample rate.
- Analysis uses the intersection of voiced frames detected in A, B and C.
- Loudness-normalized analysis targets -23.0 LUFS per file.
- Raw waveform correlation is intentionally not used.

## Integrity

| Sample | Hz | Frames | Seconds | Common voiced frames |
|---|---|---|---|---|
| r13_00_15s2 | 44100 | 420864 | 9.543 | 787 |
| r13_00_19s3 | 44100 | 529408 | 12.005 | 941 |
| r13_00_23s2 | 44100 | 392704 | 8.905 | 729 |

## Method

- H1-H2: uncorrected local spectral peaks around F0 and 2F0, 60 ms Hann frames.
- CPP: frame cepstral peak above a fitted quefrency baseline near the detected pitch period.
- HNR: Praat cross-correlation harmonicity on the same 10 ms frame grid.
- Spectral tilt: least-squares slope of log magnitude from 0.5-5 kHz.
- Aperiodic ratios: non-harmonic power / total power after excluding bands around F0 harmonics; values closer to 0 dB mean more noise.
- F0 velocity and acceleration exclude transitions across unvoiced gaps.
- Attack/release slopes use the first/last up-to-100 ms of common voiced segments.
- LUFS uses torchaudio's ITU-R BS.1770 implementation; RMS is measured only on common voiced frames.

## Raw audio: Per-sample voiced metrics

| Sample | Path | H1-H2 | CPP | HNR | Spectral tilt | Aperiodic ratio 1-5k | Aperiodic ratio 2-8k | Voiced F0 cents std | Abs F0 velocity | Abs F0 acceleration | Attack slope | Release slope | Voiced RMS | Integrated loudness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r13_00_15s2 | A | 12.295 | 1.702 | 24.028 | -2.970 | -17.366 | -9.210 | 276.003 | 4.778 | 2.326 | 69.051 | -159.584 | -14.614 | -15.264 |
| r13_00_15s2 | B | 12.235 | 1.733 | 21.673 | -2.905 | -15.749 | -8.431 | 282.389 | 6.742 | 3.469 | 52.471 | -144.984 | -13.612 | -14.240 |
| r13_00_15s2 | C | 12.111 | 1.725 | 20.678 | -2.848 | -15.432 | -8.045 | 284.989 | 7.404 | 4.104 | 60.331 | -140.826 | -13.827 | -14.356 |
| r13_00_19s3 | A | 5.818 | 1.869 | 21.986 | -2.214 | -19.777 | -11.091 | 359.893 | 5.155 | 2.380 | 69.447 | -134.049 | -14.745 | -15.164 |
| r13_00_19s3 | B | 5.985 | 1.927 | 21.253 | -2.172 | -19.255 | -10.567 | 365.538 | 6.148 | 3.122 | 54.347 | -136.849 | -13.688 | -14.159 |
| r13_00_19s3 | C | 5.975 | 1.930 | 20.070 | -2.077 | -18.742 | -10.107 | 366.746 | 7.737 | 3.851 | 54.185 | -127.118 | -13.858 | -14.333 |
| r13_00_23s2 | A | 10.783 | 1.854 | 23.884 | -2.861 | -18.835 | -9.890 | 273.027 | 4.649 | 2.269 | 60.540 | -128.657 | -15.012 | -15.824 |
| r13_00_23s2 | B | 10.639 | 1.826 | 22.336 | -2.812 | -17.871 | -9.315 | 278.124 | 5.859 | 3.435 | 49.357 | -122.372 | -14.162 | -14.812 |
| r13_00_23s2 | C | 10.825 | 1.808 | 21.591 | -2.679 | -17.424 | -8.749 | 275.851 | 6.213 | 3.913 | 50.948 | -118.475 | -14.263 | -14.953 |

## Raw audio: Pooled statistics

| Metric | Unit | Path | N | Mean | Median | P25 | P75 | P90 |
|---|---|---|---|---|---|---|---|---|
| H1-H2 | dB | A | 2457 | 9.396 | 9.899 | 3.255 | 14.302 | 19.451 |
| H1-H2 | dB | B | 2457 | 9.442 | 9.846 | 3.287 | 14.506 | 19.639 |
| H1-H2 | dB | C | 2457 | 9.513 | 9.912 | 3.391 | 14.610 | 19.734 |
| CPP | dB | A | 2457 | 1.769 | 1.801 | 1.432 | 2.137 | 2.468 |
| CPP | dB | B | 2457 | 1.787 | 1.826 | 1.439 | 2.182 | 2.510 |
| CPP | dB | C | 2457 | 1.794 | 1.817 | 1.462 | 2.189 | 2.534 |
| HNR | dB | A | 2393 | 21.891 | 23.169 | 17.093 | 26.661 | 28.636 |
| HNR | dB | B | 2393 | 20.915 | 21.679 | 16.485 | 25.569 | 27.741 |
| HNR | dB | C | 2392 | 20.401 | 20.710 | 16.043 | 24.869 | 27.280 |
| Spectral tilt | dB/kHz | A | 2457 | -2.631 | -2.660 | -3.536 | -1.778 | -0.931 |
| Spectral tilt | dB/kHz | B | 2457 | -2.618 | -2.585 | -3.531 | -1.732 | -0.841 |
| Spectral tilt | dB/kHz | C | 2457 | -2.594 | -2.517 | -3.512 | -1.692 | -0.855 |
| Aperiodic ratio 1-5k | dB | A | 2457 | -17.296 | -18.850 | -23.330 | -10.809 | -4.435 |
| Aperiodic ratio 1-5k | dB | B | 2457 | -16.737 | -17.885 | -23.054 | -10.450 | -3.932 |
| Aperiodic ratio 1-5k | dB | C | 2457 | -16.431 | -17.300 | -22.797 | -10.195 | -3.902 |
| Aperiodic ratio 2-8k | dB | A | 2457 | -9.457 | -9.836 | -13.441 | -4.610 | -1.975 |
| Aperiodic ratio 2-8k | dB | B | 2457 | -9.036 | -9.311 | -13.172 | -3.971 | -1.813 |
| Aperiodic ratio 2-8k | dB | C | 2457 | -8.703 | -8.869 | -12.744 | -3.720 | -1.855 |
| Voiced F0 cents std | cents | A | 3 | 302.974 | 276.003 | 274.515 | 317.948 | 343.115 |
| Voiced F0 cents std | cents | B | 3 | 308.683 | 282.389 | 280.256 | 323.963 | 348.908 |
| Voiced F0 cents std | cents | C | 3 | 309.195 | 284.989 | 280.420 | 325.867 | 350.395 |
| Abs F0 velocity | cents/frame | A | 2393 | 10.701 | 4.855 | 1.883 | 12.708 | 24.451 |
| Abs F0 velocity | cents/frame | B | 2393 | 12.804 | 6.252 | 2.557 | 15.427 | 30.722 |
| Abs F0 velocity | cents/frame | C | 2393 | 13.680 | 7.166 | 2.786 | 16.484 | 30.836 |
| Abs F0 acceleration | cents/frame^2 | A | 2341 | 6.561 | 2.334 | 1.050 | 4.852 | 10.065 |
| Abs F0 acceleration | cents/frame^2 | B | 2341 | 7.765 | 3.339 | 1.438 | 7.194 | 13.514 |
| Abs F0 acceleration | cents/frame^2 | C | 2341 | 8.803 | 3.948 | 1.735 | 8.013 | 14.095 |
| Attack slope | dB/s | A | 52 | 74.116 | 65.791 | 52.317 | 93.839 | 121.068 |
| Attack slope | dB/s | B | 52 | 60.744 | 51.992 | 37.043 | 77.759 | 104.259 |
| Attack slope | dB/s | C | 52 | 59.559 | 55.104 | 36.222 | 76.933 | 101.803 |
| Release slope | dB/s | A | 52 | -147.037 | -134.932 | -180.646 | -108.957 | -93.448 |
| Release slope | dB/s | B | 52 | -146.434 | -136.950 | -173.262 | -106.405 | -93.529 |
| Release slope | dB/s | C | 52 | -142.814 | -132.135 | -169.147 | -104.468 | -89.611 |
| Voiced RMS | dBFS | A | 2457 | -15.125 | -14.820 | -16.289 | -13.207 | -11.800 |
| Voiced RMS | dBFS | B | 2457 | -14.080 | -13.858 | -15.312 | -12.208 | -10.785 |
| Voiced RMS | dBFS | C | 2457 | -14.215 | -14.005 | -15.540 | -12.355 | -10.929 |
| Integrated loudness | LUFS | A | 3 | -15.417 | -15.264 | -15.544 | -15.214 | -15.184 |
| Integrated loudness | LUFS | B | 3 | -14.404 | -14.240 | -14.526 | -14.199 | -14.175 |
| Integrated loudness | LUFS | C | 3 | -14.547 | -14.356 | -14.655 | -14.344 | -14.337 |

## Raw audio: Paired deltas

Delta is second path minus first path on matched frames/segments.

| Metric | Unit | Comparison | N | Mean delta | Median | P25 | P75 | P90 |
|---|---|---|---|---|---|---|---|---|
| H1-H2 | dB | B-A | 2457 | 0.046 | 0.069 | -0.397 | 0.476 | 0.932 |
| H1-H2 | dB | C-A | 2457 | 0.117 | 0.118 | -0.346 | 0.556 | 1.116 |
| H1-H2 | dB | C-B | 2457 | 0.071 | 0.032 | -0.213 | 0.309 | 0.682 |
| CPP | dB | B-A | 2457 | 0.019 | 0.015 | -0.148 | 0.172 | 0.356 |
| CPP | dB | C-A | 2457 | 0.025 | 0.018 | -0.149 | 0.186 | 0.369 |
| CPP | dB | C-B | 2457 | 0.007 | 0.002 | -0.123 | 0.136 | 0.275 |
| HNR | dB | B-A | 2386 | -0.989 | -0.758 | -2.125 | 0.271 | 1.269 |
| HNR | dB | C-A | 2386 | -1.511 | -1.065 | -2.856 | 0.240 | 1.436 |
| HNR | dB | C-B | 2387 | -0.522 | -0.329 | -1.118 | 0.301 | 0.947 |
| Spectral tilt | dB/kHz | B-A | 2457 | 0.013 | 0.013 | -0.264 | 0.300 | 0.544 |
| Spectral tilt | dB/kHz | C-A | 2457 | 0.037 | 0.039 | -0.246 | 0.321 | 0.584 |
| Spectral tilt | dB/kHz | C-B | 2457 | 0.024 | 0.026 | -0.207 | 0.252 | 0.453 |
| Aperiodic ratio 1-5k | dB | B-A | 2457 | 0.560 | 0.377 | -0.640 | 1.535 | 3.440 |
| Aperiodic ratio 1-5k | dB | C-A | 2457 | 0.865 | 0.420 | -0.644 | 1.919 | 4.385 |
| Aperiodic ratio 1-5k | dB | C-B | 2457 | 0.305 | 0.170 | -0.604 | 1.035 | 2.202 |
| Aperiodic ratio 2-8k | dB | B-A | 2457 | 0.422 | 0.253 | -0.715 | 1.368 | 2.998 |
| Aperiodic ratio 2-8k | dB | C-A | 2457 | 0.754 | 0.381 | -0.614 | 1.791 | 3.924 |
| Aperiodic ratio 2-8k | dB | C-B | 2457 | 0.332 | 0.149 | -0.493 | 1.025 | 2.190 |
| Voiced F0 cents std | cents | B-A | 3 | 5.709 | 5.645 | 5.371 | 6.016 | 6.238 |
| Voiced F0 cents std | cents | C-A | 3 | 6.221 | 6.853 | 4.839 | 7.920 | 8.559 |
| Voiced F0 cents std | cents | C-B | 3 | 0.512 | 1.208 | -0.532 | 1.904 | 2.321 |
| Abs F0 velocity | cents/frame | B-A | 2393 | 2.103 | 0.997 | -0.990 | 4.324 | 9.828 |
| Abs F0 velocity | cents/frame | C-A | 2393 | 2.979 | 1.515 | -0.977 | 5.680 | 11.703 |
| Abs F0 velocity | cents/frame | C-B | 2393 | 0.876 | 0.405 | -1.272 | 2.413 | 4.909 |
| Abs F0 acceleration | cents/frame^2 | B-A | 2341 | 1.204 | 0.696 | -0.887 | 3.136 | 7.046 |
| Abs F0 acceleration | cents/frame^2 | C-A | 2341 | 2.241 | 1.212 | -0.557 | 4.102 | 8.207 |
| Abs F0 acceleration | cents/frame^2 | C-B | 2341 | 1.037 | 0.349 | -1.201 | 2.190 | 4.750 |
| Attack slope | dB/s | B-A | 52 | -13.372 | -13.331 | -17.676 | -9.337 | -3.432 |
| Attack slope | dB/s | C-A | 52 | -14.557 | -15.571 | -18.924 | -9.877 | -5.037 |
| Attack slope | dB/s | C-B | 52 | -1.185 | -1.810 | -4.700 | 0.832 | 4.307 |
| Release slope | dB/s | B-A | 52 | 0.604 | 4.262 | -5.252 | 9.693 | 15.977 |
| Release slope | dB/s | C-A | 52 | 4.224 | 6.795 | -0.179 | 14.050 | 17.086 |
| Release slope | dB/s | C-B | 52 | 3.620 | 3.306 | 0.963 | 5.930 | 9.415 |
| Voiced RMS | dBFS | B-A | 2457 | 1.045 | 0.927 | 0.625 | 1.443 | 1.833 |
| Voiced RMS | dBFS | C-A | 2457 | 0.910 | 0.789 | 0.373 | 1.408 | 1.861 |
| Voiced RMS | dBFS | C-B | 2457 | -0.135 | -0.145 | -0.286 | -0.001 | 0.143 |
| Integrated loudness | LUFS | B-A | 3 | 1.014 | 1.012 | 1.008 | 1.018 | 1.022 |
| Integrated loudness | LUFS | C-A | 3 | 0.870 | 0.871 | 0.851 | 0.890 | 0.901 |
| Integrated loudness | LUFS | C-B | 3 | -0.144 | -0.141 | -0.157 | -0.128 | -0.121 |

## Loudness normalized (-23 LUFS): Per-sample voiced metrics

| Sample | Path | H1-H2 | CPP | HNR | Spectral tilt | Aperiodic ratio 1-5k | Aperiodic ratio 2-8k | Voiced F0 cents std | Abs F0 velocity | Abs F0 acceleration | Attack slope | Release slope | Voiced RMS | Integrated loudness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r13_00_15s2 | A | 12.295 | 1.702 | 24.028 | -2.970 | -17.366 | -9.210 | 276.003 | 4.778 | 2.326 | 69.051 | -159.584 | -22.350 | -23.000 |
| r13_00_15s2 | B | 12.235 | 1.733 | 21.673 | -2.905 | -15.749 | -8.431 | 282.389 | 6.742 | 3.469 | 52.471 | -144.984 | -22.373 | -23.000 |
| r13_00_15s2 | C | 12.111 | 1.725 | 20.678 | -2.848 | -15.432 | -8.045 | 284.989 | 7.404 | 4.104 | 60.331 | -140.826 | -22.471 | -23.000 |
| r13_00_19s3 | A | 5.818 | 1.869 | 21.986 | -2.214 | -19.777 | -11.091 | 359.893 | 5.155 | 2.380 | 69.447 | -134.049 | -22.581 | -23.000 |
| r13_00_19s3 | B | 5.985 | 1.927 | 21.253 | -2.172 | -19.255 | -10.567 | 365.538 | 6.148 | 3.122 | 54.347 | -136.849 | -22.529 | -23.000 |
| r13_00_19s3 | C | 5.975 | 1.930 | 20.070 | -2.077 | -18.742 | -10.107 | 366.746 | 7.737 | 3.851 | 54.185 | -127.118 | -22.525 | -23.000 |
| r13_00_23s2 | A | 10.783 | 1.854 | 23.884 | -2.861 | -18.835 | -9.890 | 273.027 | 4.649 | 2.269 | 60.540 | -128.657 | -22.187 | -23.000 |
| r13_00_23s2 | B | 10.639 | 1.826 | 22.336 | -2.812 | -17.871 | -9.315 | 278.124 | 5.859 | 3.435 | 49.357 | -122.372 | -22.350 | -23.000 |
| r13_00_23s2 | C | 10.825 | 1.808 | 21.591 | -2.679 | -17.424 | -8.749 | 275.851 | 6.213 | 3.913 | 50.948 | -118.475 | -22.309 | -23.000 |

## Loudness normalized (-23 LUFS): Pooled statistics

| Metric | Unit | Path | N | Mean | Median | P25 | P75 | P90 |
|---|---|---|---|---|---|---|---|---|
| H1-H2 | dB | A | 2457 | 9.396 | 9.899 | 3.255 | 14.302 | 19.451 |
| H1-H2 | dB | B | 2457 | 9.442 | 9.846 | 3.287 | 14.506 | 19.639 |
| H1-H2 | dB | C | 2457 | 9.513 | 9.912 | 3.391 | 14.610 | 19.734 |
| CPP | dB | A | 2457 | 1.769 | 1.801 | 1.432 | 2.137 | 2.468 |
| CPP | dB | B | 2457 | 1.787 | 1.826 | 1.439 | 2.182 | 2.510 |
| CPP | dB | C | 2457 | 1.794 | 1.817 | 1.462 | 2.189 | 2.534 |
| HNR | dB | A | 2393 | 21.891 | 23.169 | 17.093 | 26.661 | 28.636 |
| HNR | dB | B | 2393 | 20.915 | 21.679 | 16.485 | 25.569 | 27.741 |
| HNR | dB | C | 2392 | 20.401 | 20.710 | 16.043 | 24.869 | 27.280 |
| Spectral tilt | dB/kHz | A | 2457 | -2.631 | -2.660 | -3.536 | -1.778 | -0.931 |
| Spectral tilt | dB/kHz | B | 2457 | -2.618 | -2.585 | -3.531 | -1.732 | -0.841 |
| Spectral tilt | dB/kHz | C | 2457 | -2.594 | -2.517 | -3.512 | -1.692 | -0.855 |
| Aperiodic ratio 1-5k | dB | A | 2457 | -17.296 | -18.850 | -23.330 | -10.809 | -4.435 |
| Aperiodic ratio 1-5k | dB | B | 2457 | -16.737 | -17.885 | -23.054 | -10.450 | -3.932 |
| Aperiodic ratio 1-5k | dB | C | 2457 | -16.431 | -17.300 | -22.797 | -10.195 | -3.902 |
| Aperiodic ratio 2-8k | dB | A | 2457 | -9.457 | -9.836 | -13.441 | -4.610 | -1.975 |
| Aperiodic ratio 2-8k | dB | B | 2457 | -9.036 | -9.311 | -13.172 | -3.971 | -1.813 |
| Aperiodic ratio 2-8k | dB | C | 2457 | -8.703 | -8.869 | -12.744 | -3.720 | -1.855 |
| Voiced F0 cents std | cents | A | 3 | 302.974 | 276.003 | 274.515 | 317.948 | 343.115 |
| Voiced F0 cents std | cents | B | 3 | 308.683 | 282.389 | 280.256 | 323.963 | 348.908 |
| Voiced F0 cents std | cents | C | 3 | 309.195 | 284.989 | 280.420 | 325.867 | 350.395 |
| Abs F0 velocity | cents/frame | A | 2393 | 10.701 | 4.855 | 1.883 | 12.708 | 24.451 |
| Abs F0 velocity | cents/frame | B | 2393 | 12.804 | 6.252 | 2.557 | 15.427 | 30.722 |
| Abs F0 velocity | cents/frame | C | 2393 | 13.680 | 7.166 | 2.786 | 16.484 | 30.836 |
| Abs F0 acceleration | cents/frame^2 | A | 2341 | 6.561 | 2.334 | 1.050 | 4.852 | 10.065 |
| Abs F0 acceleration | cents/frame^2 | B | 2341 | 7.765 | 3.339 | 1.438 | 7.194 | 13.514 |
| Abs F0 acceleration | cents/frame^2 | C | 2341 | 8.803 | 3.948 | 1.735 | 8.013 | 14.095 |
| Attack slope | dB/s | A | 52 | 74.116 | 65.791 | 52.317 | 93.839 | 121.068 |
| Attack slope | dB/s | B | 52 | 60.744 | 51.992 | 37.043 | 77.759 | 104.259 |
| Attack slope | dB/s | C | 52 | 59.559 | 55.104 | 36.222 | 76.933 | 101.803 |
| Release slope | dB/s | A | 52 | -147.037 | -134.932 | -180.646 | -108.957 | -93.448 |
| Release slope | dB/s | B | 52 | -146.434 | -136.950 | -173.262 | -106.405 | -93.529 |
| Release slope | dB/s | C | 52 | -142.814 | -132.135 | -169.147 | -104.468 | -89.611 |
| Voiced RMS | dBFS | A | 2457 | -22.733 | -22.350 | -23.912 | -20.714 | -19.495 |
| Voiced RMS | dBFS | B | 2457 | -22.701 | -22.409 | -23.889 | -20.842 | -19.439 |
| Voiced RMS | dBFS | C | 2457 | -22.691 | -22.421 | -23.933 | -20.867 | -19.441 |
| Integrated loudness | LUFS | A | 3 | -23.000 | -23.000 | -23.000 | -23.000 | -23.000 |
| Integrated loudness | LUFS | B | 3 | -23.000 | -23.000 | -23.000 | -23.000 | -23.000 |
| Integrated loudness | LUFS | C | 3 | -23.000 | -23.000 | -23.000 | -23.000 | -23.000 |

## Loudness normalized (-23 LUFS): Paired deltas

Delta is second path minus first path on matched frames/segments.

| Metric | Unit | Comparison | N | Mean delta | Median | P25 | P75 | P90 |
|---|---|---|---|---|---|---|---|---|
| H1-H2 | dB | B-A | 2457 | 0.046 | 0.069 | -0.397 | 0.476 | 0.932 |
| H1-H2 | dB | C-A | 2457 | 0.117 | 0.118 | -0.346 | 0.556 | 1.116 |
| H1-H2 | dB | C-B | 2457 | 0.071 | 0.032 | -0.213 | 0.309 | 0.682 |
| CPP | dB | B-A | 2457 | 0.019 | 0.015 | -0.148 | 0.172 | 0.356 |
| CPP | dB | C-A | 2457 | 0.025 | 0.018 | -0.149 | 0.186 | 0.369 |
| CPP | dB | C-B | 2457 | 0.007 | 0.002 | -0.123 | 0.136 | 0.275 |
| HNR | dB | B-A | 2386 | -0.989 | -0.758 | -2.125 | 0.271 | 1.269 |
| HNR | dB | C-A | 2386 | -1.511 | -1.065 | -2.856 | 0.240 | 1.436 |
| HNR | dB | C-B | 2387 | -0.522 | -0.329 | -1.118 | 0.301 | 0.947 |
| Spectral tilt | dB/kHz | B-A | 2457 | 0.013 | 0.013 | -0.264 | 0.300 | 0.544 |
| Spectral tilt | dB/kHz | C-A | 2457 | 0.037 | 0.039 | -0.246 | 0.321 | 0.584 |
| Spectral tilt | dB/kHz | C-B | 2457 | 0.024 | 0.026 | -0.207 | 0.252 | 0.453 |
| Aperiodic ratio 1-5k | dB | B-A | 2457 | 0.560 | 0.377 | -0.640 | 1.535 | 3.440 |
| Aperiodic ratio 1-5k | dB | C-A | 2457 | 0.865 | 0.420 | -0.644 | 1.919 | 4.385 |
| Aperiodic ratio 1-5k | dB | C-B | 2457 | 0.305 | 0.170 | -0.604 | 1.035 | 2.202 |
| Aperiodic ratio 2-8k | dB | B-A | 2457 | 0.422 | 0.253 | -0.715 | 1.368 | 2.998 |
| Aperiodic ratio 2-8k | dB | C-A | 2457 | 0.754 | 0.381 | -0.614 | 1.791 | 3.924 |
| Aperiodic ratio 2-8k | dB | C-B | 2457 | 0.332 | 0.149 | -0.493 | 1.025 | 2.190 |
| Voiced F0 cents std | cents | B-A | 3 | 5.709 | 5.645 | 5.371 | 6.016 | 6.238 |
| Voiced F0 cents std | cents | C-A | 3 | 6.221 | 6.853 | 4.839 | 7.920 | 8.559 |
| Voiced F0 cents std | cents | C-B | 3 | 0.512 | 1.208 | -0.532 | 1.904 | 2.321 |
| Abs F0 velocity | cents/frame | B-A | 2393 | 2.103 | 0.997 | -0.990 | 4.324 | 9.828 |
| Abs F0 velocity | cents/frame | C-A | 2393 | 2.979 | 1.515 | -0.977 | 5.680 | 11.703 |
| Abs F0 velocity | cents/frame | C-B | 2393 | 0.876 | 0.405 | -1.272 | 2.413 | 4.909 |
| Abs F0 acceleration | cents/frame^2 | B-A | 2341 | 1.204 | 0.696 | -0.887 | 3.136 | 7.046 |
| Abs F0 acceleration | cents/frame^2 | C-A | 2341 | 2.241 | 1.212 | -0.557 | 4.102 | 8.207 |
| Abs F0 acceleration | cents/frame^2 | C-B | 2341 | 1.037 | 0.349 | -1.201 | 2.190 | 4.750 |
| Attack slope | dB/s | B-A | 52 | -13.372 | -13.331 | -17.676 | -9.337 | -3.432 |
| Attack slope | dB/s | C-A | 52 | -14.557 | -15.571 | -18.924 | -9.877 | -5.037 |
| Attack slope | dB/s | C-B | 52 | -1.185 | -1.810 | -4.700 | 0.832 | 4.307 |
| Release slope | dB/s | B-A | 52 | 0.604 | 4.262 | -5.252 | 9.693 | 15.977 |
| Release slope | dB/s | C-A | 52 | 4.224 | 6.795 | -0.179 | 14.050 | 17.086 |
| Release slope | dB/s | C-B | 52 | 3.620 | 3.306 | 0.963 | 5.930 | 9.415 |
| Voiced RMS | dBFS | B-A | 2457 | 0.032 | -0.086 | -0.388 | 0.429 | 0.820 |
| Voiced RMS | dBFS | C-A | 2457 | 0.042 | -0.080 | -0.490 | 0.547 | 1.002 |
| Voiced RMS | dBFS | C-B | 2457 | 0.010 | -0.001 | -0.141 | 0.145 | 0.287 |
| Integrated loudness | LUFS | B-A | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Integrated loudness | LUFS | C-A | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Integrated loudness | LUFS | C-B | 3 | -0.000 | 0.000 | -0.000 | 0.000 | 0.000 |

## Interpretation

### A -> B

- Supports increased breathiness: HNR falls -0.758 dB median (-0.989 dB mean); both aperiodic ratios move toward 0 dB, indicating more non-harmonic energy.
- No material change: H1-H2 and CPP remain nearly flat.
- Weak opposite cue: spectral tilt becomes only slightly flatter, not steeper.

### A -> C

- Supports increased breathiness: HNR falls -1.065 dB median (-1.511 dB mean).
- Additional support: 1-5 kHz aperiodic ratio rises +0.865 dB mean; 2-8 kHz rises +0.754 dB mean.
- No material change: H1-H2 is +0.118 dB median and CPP is +0.018 dB median.
- Weak opposite cue: spectral tilt flattens by +0.039 dB/kHz instead of becoming steeper.

### B -> C

- Personalized adds a small further noise/aperiodicity shift: HNR -0.329 dB median; 1-5 kHz noise ratio +0.305 dB mean.
- H1-H2, CPP and spectral tilt remain effectively unchanged, so the private stage does not create a broad spectral breathiness transformation.

## Non-breathiness checks

- C-A normalized voiced RMS median delta: -0.080 dB.
- C-A normalized LUFS median delta: +0.000 LU.
- C-A F0 velocity median delta: +1.515 cents/frame.
- C-A F0 acceleration median delta: +1.212 cents/frame^2.
- C-A attack slope median delta: -15.571 dB/s.
- C-A release slope median delta: +6.795 dB/s.

## Limitations

- Only three fixed validation clips are available, so sample-level percentiles have low statistical power.
- H1-H2 is not formant-corrected; it should be interpreted together with CPP, HNR and noise ratios.
- Frame CPP follows the cepstral-peak-over-baseline definition but is not numerically interchangeable with published Praat CPPS norms.
- The aperiodic ratios are harmonic-exclusion spectral estimates, not glottal inverse-filtering measurements.
- Lower CPP/HNR and higher noise can indicate breathiness, but excessive changes may instead indicate roughness or vocoder noise.
