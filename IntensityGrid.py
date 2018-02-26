import numpy as np

from KernelFunc import KernelFunc

from SupIntensity import SupIntensity

def IntensityGrid(para, sim_seq, mu):

	grid = np.linspace(0,sim_seq[-1],num=len(sim_seq)*1000,endpoint=False)

	intensitygrid = np.array([mu])

	for i in range(1,len(grid)):

		t_i = grid[i]

		hist_ti = sim_seq[sim_seq < t_i]

		array_inv = t_i - hist_ti

		tmp = KernelFunc(array_inv, para)

		tmp = np.sum(tmp)

#		try:

		intensitygrid = np.append(intensitygrid, tmp + mu)

#		except(TypeError):
#
#			intensitygrid = np.append(intensitygrid, mu)

	return {'intensitygrid': intensitygrid, 'grid': grid}
