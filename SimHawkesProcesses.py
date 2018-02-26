
import numpy as np
import numpy.random as rand
from KernelFunc import KernelFunc
from SupIntensity import SupIntensity

def simHP(level, para, maxjumps, taumax, Delta):

	eps = np.finfo(float).eps

	K1_Param = para

	if K1_Param['K1_Type'] == 'EXP':

		statcriter = K1_Param['EXP_statcriter']

	if K1_Param['K1_Type'] == 'PWL':

		statcriter = K1_Param['PWL_statcriter']

	if K1_Param['K1_Type'] == 'SQR':

		statcriter = K1_Param['SQR_statcriter']

	if K1_Param['K1_Type'] == 'SNS':

		statcriter = K1_Param['SNS_statcriter']

	if statcriter >= 1.:

		print('Error: The sequence could not be modeled, because the estimated kernel is not stable.')

		return np.zeros((maxjumps,))

	mu = Delta*(1-statcriter) + eps

	sim_seq = np.array([rand.exponential(1/mu)])

	n_of_jumps = 1

	time = sim_seq[0]

	n_iter = 0

	while (n_of_jumps < maxjumps) and (n_iter < 1e8):

		l = 2*taumax

		u = rand.random()

		mt = SupIntensity(para, sim_seq, mu, taumax)

		#print('mt: ' + repr(mt))

		dt = rand.exponential(1/mt)

		#dt = np.array(dt)

		#dt = dt[0]

		#print ('dt: '+repr(dt))

		#print('l: '+repr(l))
		
		#print('u: '+repr(u))

		intens_dt = SupIntensity(para, np.append(sim_seq, time+dt), mu, taumax)

		if (dt < l) and ((intens_dt/mt) < u) :

			time += dt

			sim_seq = np.append(sim_seq, time)

			n_of_jumps += 1

		else:

			time += l

		n_iter += 1





	return sim_seq
