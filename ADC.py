import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. Simulation parameters and ideal signal
# ==================================================

f_in = 1e3        
f_ref = 1e6         
duration = 3e-3    

fs = 20e3         


# Dense analog-like reference
t_ref = np.arange(0, duration, 1 / f_ref)

x_ref = np.sin(
    2 * np.pi * f_in * t_ref
)


# Actual ADC sample times
t_samp = np.arange(
    0,
    duration,
    1 / fs
)

x_samp = np.sin(
    2 * np.pi * f_in * t_samp
)


# Plot analog-like reference and samples
plt.figure()

plt.plot(
    t_ref * 1e3,
    x_ref,
    label="Analog-like reference"
)

plt.scatter(
    t_samp * 1e3,
    x_samp,
    label="ADC samples"
)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (V)")
plt.grid()
plt.legend()
plt.show()


# ==================================================
# 2. Ideal 8-bit ADC quantization
# ==================================================

bits = 8

vmin = -1.0
vmax = 1.0

levels = 2**bits

q_step = (
    vmax - vmin
) / levels


print("Number of levels:", levels)
print("LSB size:", q_step, "V")
print("LSB size:", q_step * 1e3, "mV")


# Keep input inside valid ADC code range
x_clip = np.clip(
    x_samp,
    vmin,
    vmax - q_step
)


# Convert voltage to integer ADC code
codes = np.floor(
    (x_clip - vmin) / q_step
).astype(int)


# Reconstruct voltage using center of each bin
x_quant = (
    vmin
    + (codes + 0.5) * q_step
)


# Compare unquantized and quantized samples
plt.figure()

plt.scatter(
    t_samp * 1e3,
    x_samp,
    label="Unquantized samples"
)

plt.scatter(
    t_samp * 1e3,
    x_quant,
    label="Quantized samples"
)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (V)")
plt.grid()
plt.legend()
plt.show()


# Quantization error
q_error = (
    x_quant - x_samp
)


plt.figure()

plt.plot(
    t_samp * 1e3,
    q_error * 1e3
)

plt.xlabel("Time (ms)")
plt.ylabel("Quantization error (mV)")
plt.grid()
plt.show()


print(
    "Minimum quantization error:",
    q_error.min() * 1e3,
    "mV"
)

print(
    "Maximum quantization error:",
    q_error.max() * 1e3,
    "mV"
)


# ==================================================
# 3. Two-channel time interleaving
# ==================================================

# Channel A takes even samples
t_A = t_samp[0::2]
x_A = x_samp[0::2]

# Channel B takes odd samples
t_B = t_samp[1::2]
x_B = x_samp[1::2]


print(
    "First five A times (ms):",
    t_A[:5] * 1e3
)

print(
    "First five B times (ms):",
    t_B[:5] * 1e3
)


# ==================================================
# 4. Gain mismatch experiment
# ==================================================

gain_B = 1.02

x_B_bad = (
    gain_B * x_B
)


# Rebuild interleaved stream
x_interleaved_gain = x_samp.copy()

x_interleaved_gain[1::2] = (
    x_B_bad
)


plt.figure()

plt.scatter(
    t_samp * 1e3,
    x_samp,
    label="Ideal samples"
)

plt.scatter(
    t_samp * 1e3,
    x_interleaved_gain,
    label="2% B-channel gain error"
)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (V)")
plt.grid()
plt.legend()
plt.show()


gain_error = (
    x_interleaved_gain
    - x_samp
)


# ==================================================
# 5. Offset mismatch experiment
# ==================================================

offset_B = 0.020     # +20 mV

x_B_offset = (
    x_B + offset_B
)


# Rebuild interleaved stream
x_interleaved_offset = (
    x_samp.copy()
)

x_interleaved_offset[1::2] = (
    x_B_offset
)


offset_error = (
    x_interleaved_offset
    - x_samp
)


plt.figure()

plt.plot(
    t_samp * 1e3,
    offset_error * 1e3
)

plt.xlabel("Time (ms)")
plt.ylabel("Offset error (mV)")
plt.grid()
plt.show()


# ==================================================
# 6. FFT of offset-mismatched ADC
# ==================================================

plt.close("all")


N = len(
    x_interleaved_offset
)


X_offset = np.fft.rfft(
    x_interleaved_offset
)

freqs = np.fft.rfftfreq(
    N,
    d=1 / fs
)

mag_offset = np.abs(
    X_offset
)


plt.figure()

plt.plot(
    freqs / 1e3,
    mag_offset
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()


# ==================================================
# 7. Digital offset calibration
# ==================================================

# Estimate B-channel offset from its mean
offset_est = np.mean(
    x_B_offset
)

print(
    "Estimated B offset:",
    offset_est
)


# Correct B channel
x_B_cal = (
    x_B_offset
    - offset_est
)


print(
    "Calibrated B mean:",
    np.mean(x_B_cal)
)


# Rebuild calibrated interleaved stream
x_interleaved_cal = (
    x_samp.copy()
)

x_interleaved_cal[1::2] = (
    x_B_cal
)


# FFT after calibration
X_cal = np.fft.rfft(
    x_interleaved_cal
)

mag_cal = np.abs(
    X_cal
)


plt.figure()

plt.plot(
    freqs / 1e3,
    mag_cal
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()


# Locate 1 kHz signal and 10 kHz offset spur
idx_1k = np.argmin(
    np.abs(freqs - 1e3)
)

idx_10k = np.argmin(
    np.abs(freqs - 10e3)
)


print(
    "1 kHz magnitude:",
    mag_cal[idx_1k]
)

print(
    "10 kHz magnitude:",
    mag_cal[idx_10k]
)


# ==================================================
# 8. Before / after calibration comparison
# ==================================================

mag_bad = np.abs(
    np.fft.rfft(
        x_interleaved_offset
    )
)

mag_cal = np.abs(
    np.fft.rfft(
        x_interleaved_cal
    )
)


spur_before = (
    mag_bad[idx_10k]
)

spur_after = (
    mag_cal[idx_10k]
)


print(
    "10 kHz spur before:",
    spur_before
)

print(
    "10 kHz spur after:",
    spur_after
)


plt.figure()

plt.plot(
    freqs / 1e3,
    mag_bad,
    label="Before calibration"
)

plt.plot(
    freqs / 1e3,
    mag_cal,
    label="After calibration"
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.grid()
plt.legend()
plt.show()


# ==================================================
# 9. Timing skew experiment
# ==================================================

timing_skew = 2e-6      # B samples 2 us late


# Shift B-channel sampling instants
t_B_skew = (
    t_B + timing_skew
)


# Sample actual input at shifted times
x_B_skew = np.sin(
    2
    * np.pi
    * f_in
    * t_B_skew
)


# Error caused by timing skew
timing_error_B = (
    x_B_skew - x_B
)


plt.figure()

plt.plot(
    t_B * 1e3,
    timing_error_B
)

plt.xlabel("Ideal B sample time (ms)")
plt.ylabel("Timing-skew error (V)")
plt.grid()
plt.show()


# Rebuild full interleaved stream
x_interleaved_skew = (
    x_samp.copy()
)

x_interleaved_skew[1::2] = (
    x_B_skew
)


# ==================================================
# 10. FFT of timing-skewed ADC
# ==================================================

plt.close("all")


X_skew = np.fft.rfft(
    x_interleaved_skew
)

freqs_skew = np.fft.rfftfreq(
    len(x_interleaved_skew),
    d=1 / fs
)

mag_skew = np.abs(
    X_skew
)


plt.figure()

plt.plot(
    freqs_skew / 1e3,
    mag_skew
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")

plt.xlim(
    0,
    fs / 2 / 1e3
)

plt.grid()
plt.show()


# ==================================================
# 11. Timing-skew frequency dependence test
# ==================================================

# Test at 2 kHz while keeping the same 2 us skew
f_test = 2e3


x_B_2k = np.sin(
    2
    * np.pi
    * f_test
    * t_B
)


x_B_skew_2k = np.sin(
    2
    * np.pi
    * f_test
    * (t_B + timing_skew)
)


timing_error_2k = (
    x_B_skew_2k
    - x_B_2k
)


print(
    "1 kHz max error:",
    np.max(
        np.abs(timing_error_B)
    )
)

print(
    "2 kHz max error:",
    np.max(
        np.abs(timing_error_2k)
    )
) 



# the most important result: the FFT before and after offset calibration.

mag_bad = np.abs(np.fft.rfft(x_interleaved_offset))
mag_cal = np.abs(np.fft.rfft(x_interleaved_cal))

plt.figure()

plt.plot(freqs / 1e3, mag_bad, label="Before calibration")
plt.plot(freqs / 1e3, mag_cal, label="After calibration")

plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude")
plt.title("Offset Calibration: Before vs After")
plt.grid()
plt.legend()

plt.savefig(
    "results/offset_calibration_fft.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# timing-skew spectrum


mag_skew_norm = mag_skew / np.max(mag_skew)
mag_skew_db = 20 * np.log10(np.maximum(mag_skew_norm, 1e-12))

plt.figure()
plt.plot(freqs_skew / 1e3, mag_skew_db)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude (dB)")
plt.title("Spectrum with Channel B Timing Skew")
plt.grid()

plt.savefig(
    "results/timing_skew_fft_db.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
