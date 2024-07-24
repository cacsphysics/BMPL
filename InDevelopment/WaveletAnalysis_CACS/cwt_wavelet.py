
import matplotlib.pyplot as plt
import pywt
import load_data as ld
import cacs_library as cl
# Created by C. A. Cartagena-Sanchez


fft = cl.spectrum_wwind.spectrum_wwind

filename = "Dataset_06192024.h5"

time = ld.load_btime(filename)
# index1 = cl.tools.finding_Index_Time(time, 60)
# index2 = cl.tools.finding_Index_Time(time, 160)
data, _, _ = ld.load_B_data(filename)
# data = data[0, 10][index1:index2]
# time = time[index1:index2]

data = data[0, 10]
# scale_powers = np.geomspace(6, 15, num=2000)
# scales = 2 ** (scale_powers)
scales = np.arange(6, 1024)
scales = np.append(scales, np.geomspace(1025, 2048, num=100))

print(scales)
print(time[0])

waveletname = "morl"
power_list = []
for num in range(30):
    coeffs, freqeuncies = pywt.cwt(
        data,
        scales,
        wavelet=waveletname,
        sampling_period=time[1] - time[0],
        method="fft",
    )

    power_list.append(np.abs(coeffs) ** 2)

freqeuncies = freqeuncies * 1e-6
power = np.asarray(power_list)
power = np.average(power, axis=0)
print(power.shape)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.contourf(time[1:] * 1e6, np.log2(freqeuncies), np.log2(power), cmap=plt.cm.jet)

# ax.set_title(title, fontsize=20)
# ax.set_ylabel(ylabel, fontsize=18)
ax.set_ylabel("Log-2 [Frequencies(MHz)]", fontsize=18)
ax.set_xlabel(r"Time ($\mu s$)", fontsize=18)


yticks = 2 ** np.arange(
    np.ceil(np.log2(freqeuncies.min())), np.ceil(np.log2(freqeuncies.max()))
)
ax.set_yticks(np.log2(yticks))
ax.set_yticklabels(yticks)
ax.invert_yaxis()
ylim = ax.get_ylim()
# ax.set_ylim(ylim[0], -1)


# cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
fig.colorbar(im, orientation="vertical")
plt.savefig("Figures/wavelet_p1_0619v6.pdf")
plt.savefig("Figures/wavelet_p1_0619v6.png")
# plt.show()


plt.close()
