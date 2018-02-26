
import numpy as np

def KernelFunc(vector, para):

	K1_Param = para

	if K1_Param['K1_Type'] == 'EXP':

		coeffs = K1_Param['EXP_coeffs']

		alpha = coeffs[0]

		beta = coeffs[1]

		return alpha*np.exp(-beta*vector)

	if K1_Param['K1_Type'] == 'PWL':

		coeffs = K1_Param['PWL_coeffs']

		K = coeffs[0]

		c = coeffs[1]

		p = coeffs[2]

		return K*np.power(vector+c,-p)

	if K1_Param['K1_Type'] == 'SQR':

		coeffs = K1_Param['SQR_coeffs']

		B = coeffs[0]

		L = coeffs[1]

		vector = np.array(vector)

		#print('L: '+repr(L))

		#print('vector before thresholding: ' + repr(vector))

		# def SQR(vector, B, L):

		vector = [0 if a_ > L else B for a_ in vector]

		#vector[vector > L] = 0

		#vector[vector <= L] = B

		#print('vector: ' + repr(vector))

		return vector

	if K1_Param['K1_Type'] == 'SNS':

		coeffs = K1_Param['SNS_coeffs']

		A = coeffs[0]

		omega = coeffs[1]

		# def SNS(vector, A, omega):

		vector[vector < np.pi/omega] = A*np.sin(omega*vector[vector < np.pi/omega])

		vector[vector >= np.pi/omega] = 0

		return vector

	if K1_Param['K1_Type']=='EXP+PWL':

		coeffs = K1_Param['EXPplusPWL_coeffs']

		alpha = coeffs[0]

		beta = coeffs[1]

		K = coeffs[2]

		c = coeffs[3]

		p = coeffs[4]

		return alpha*np.exp(-beta*vector) + K*np.power(vector+c,-p)

	if K1_Param['K1_Type']=='EXP+SNS':

		coeffs = K1_Param['EXPplusSNS_coeffs']

		alpha = coeffs[0]

		beta = coeffs[1]

		A = coeffs[2]

		omega = coeffs[3]

		vector = [alpha*np.exp(-beta*a_) if a_ > np.pi/omega else A*np.sin(omega*a_) + alpha*np.exp(-beta*a_) for a_ in vector]

		#vector = np.array(vector)

		#vector = vector + alpha*np.exp(-beta*vector)

		#vector[vector < np.pi/omega] = A*np.sin(omega*vector[vector < np.pi/omega])

		#vector[vector >= np.pi/omega] = 0

		return vector

	if K1_Param['K1_Type']=='PWLxSQR':

		coeffs = K1_Param['PWLtimesSQR_coeffs']

		KB = coeffs[0]

		c = coeffs[1]

		p = coeffs[2]

		L = coeffs[3]

		vector = [0 if a_ > L else KB*np.power(a_+c,-p) for a_ in vector]

		return vector

	if K1_Param['K1_Type']=='SNSxSNS':

		coeffs = K1_Param['SNStimesSNS_coeffs']

		A2 = coeffs[0]

		omega = coeffs[1]

		vector[vector < np.pi/omega] = A2*(np.sin(omega*vector[vector < np.pi/omega]))**2

		vector[vector >= np.pi/omega] = 0
		
		return vector
		

