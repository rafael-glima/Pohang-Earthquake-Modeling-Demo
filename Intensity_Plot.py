import matplotlib.pyplot as plt
from IntensityGrid import IntensityGrid
import numpy as np

mu = 0.4

sim_seq = np.array([0.1,0.2,0.4,0.5,0.53,0.61,0.76])*10

Param_EXP = {'K1_Type': 'EXP','EXP_coeffs':[1.,4.]}

Param_PWL = {'K1_Type': 'PWL','PWL_coeffs':[1.,1.,1.]}

Param_SQR = {'K1_Type': 'SQR', 'SQR_coeffs':[0.5,.9]}

Param_SNS = {'K1_Type': 'SNS', 'SNS_coeffs':[1.,2.3]}

Param_EXPplusPWL = {'K1_Type': 'EXP+PWL', 'EXPplusPWL_coeffs':[1.,4.,1.,1.,1.1]}

Param_EXPplusSNS = {'K1_Type': 'EXP+SNS', 'EXPplusSNS_coeffs': [1.,4.,1.,2.3]}

Param_PWLtimesSQR = {'K1_Type': 'PWLxSQR', 'PWLtimesSQR_coeffs': [1.,2.,2.,0.9]}

Param_SNStimesSNS = {'K1_Type': 'SNSxSNS', 'SNStimesSNS_coeffs': [1.,2.3]}

# EXP_Grid = IntensityGrid(Param_EXP,sim_seq,mu)

# PWL_Grid = IntensityGrid(Param_PWL,sim_seq,mu)

# SQR_Grid = IntensityGrid(Param_SQR,sim_seq,mu)

# SNS_Grid = IntensityGrid(Param_SNS,sim_seq,mu)

EXPplusPWL_Grid = IntensityGrid(Param_EXPplusPWL,sim_seq,mu)

EXPplusSNS_Grid = IntensityGrid(Param_EXPplusSNS,sim_seq,mu)

PWLtimesSQR_Grid = IntensityGrid(Param_PWLtimesSQR,sim_seq,mu)

SNStimesSNS_Grid = IntensityGrid(Param_SNStimesSNS,sim_seq,mu)

grid = EXPplusPWL_Grid['grid']

# plt.subplot(411)
# plt.plot(grid,EXP_Grid['intensitygrid'],'b-',linewidth=3)
# plt.axis([0,8,0,2])
# ax = plt.gca()
# ax.set_yticks([])
# ax.set_xticks([])
# markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
# plt.setp(markerline, 'markerfacecolor','k')
# plt.setp(baseline, 'color','k', 'linewidth', 3)

# plt.subplot(412)
# plt.plot(grid,PWL_Grid['intensitygrid'],'g-',linewidth=3)
# plt.axis([0,8,0,3.5])
# ax = plt.gca()
# ax.set_yticks([])
# ax.set_xticks([])
# markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
# plt.setp(markerline, 'markerfacecolor', 'k')
# plt.setp(baseline, 'color','k', 'linewidth', 3)

# plt.subplot(413)
# plt.plot(grid,SQR_Grid['intensitygrid'],'r-',linewidth=3)
# plt.axis([0,8,0,1.5])
# ax = plt.gca()
# ax.set_yticks([])
# ax.set_xticks([])
# markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
# plt.setp(markerline, 'markerfacecolor', 'k')
# plt.setp(baseline, 'color','k', 'linewidth', 3)

# plt.subplot(414)
# plt.plot(grid,SNS_Grid['intensitygrid'],'y-',linewidth=3)
# ax = plt.gca()
# plt.axis([0,8,0,2.5])
# ax.set_yticks([])
# ax.set_xticks([])
# markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
# plt.setp(markerline, 'markerfacecolor', 'k')
# plt.setp(baseline, 'color','k', 'linewidth', 3)

# plt.savefig('intens_4_kernels.png')

plt.subplot(411)
plt.plot(grid,EXPplusPWL_Grid['intensitygrid'],'b-',linewidth=3)
plt.axis([0,8,0,5])
ax = plt.gca()
ax.set_yticks([])
ax.set_xticks([])
markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
plt.setp(markerline, 'markerfacecolor','k')
plt.setp(baseline, 'color','k', 'linewidth', 3)

plt.subplot(412)
plt.plot(grid,EXPplusSNS_Grid['intensitygrid'],'g-',linewidth=3)
plt.axis([0,8,0,3.5])
ax = plt.gca()
ax.set_yticks([])
ax.set_xticks([])
markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
plt.setp(markerline, 'markerfacecolor', 'k')
plt.setp(baseline, 'color','k', 'linewidth', 3)

plt.subplot(413)
plt.plot(grid,PWLtimesSQR_Grid['intensitygrid'],'r-',linewidth=3)
plt.axis([0,8,0,1.])
ax = plt.gca()
ax.set_yticks([])
ax.set_xticks([])
markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
plt.setp(markerline, 'markerfacecolor', 'k')
plt.setp(baseline, 'color','k', 'linewidth', 3)

plt.subplot(414)
plt.plot(grid,SNStimesSNS_Grid['intensitygrid'],'y-',linewidth=3)
ax = plt.gca()
plt.axis([0,8,0,2.5])
ax.set_yticks([])
ax.set_xticks([])
markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
plt.setp(markerline, 'markerfacecolor', 'k')
plt.setp(baseline, 'color','k', 'linewidth', 3)

plt.savefig('intens_4_compos_kernels.png')

plt.show()
