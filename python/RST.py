import numpy as np

'''
Parameters
----------

p:
 [L,1] microphone signal in the frequency domain
f:
 frequency in Hertz
c:
 speed of sound
d:		
 distance between adjacent microphones
L:       
 Number of microphones
mbar:    
 m axis sampling interval
W:       
 length of m axis
qbar:	
 q axis sampling interval
sigma:   
 gaussian window standard deviation
Returns
------

Z:       
 [W,I] RST output 
m:       
 [W,1] m axis
q:       
 [I,1] q axis
z:       
 [L,1] microphone positions
'''


def rst(p, f, c, d, L, mbar, W, qbar, sigma):

    # Parameters
    z = np.linspace(0, d*(L-1), num=L).T                                  # [L,1] microphone positions
    m = np.linspace(-(W-1)/2*mbar, ((W-1)*mbar-((W-1)/2*mbar)), num=W).T  # [W,1] m axis
    q = np.linspace(0, z[-1], num=np.around(z[-1]/qbar) + 1).T            # [j,1] q axis
    I = np.size(q)                                                        # Number of frames

    # RST

    M, Q = np.meshgrid(m, q)
    MM, MZ = np.meshgrid(M.flatten('F'), z)
    QQ, QZ = np.meshgrid(Q.flatten('F'), z)

    # RST transformation matrix

    PSI = d*np.exp(np.divide(1j*(2*np.pi*f/c)*MZ*MM, np.sqrt(1+np.power(MM, 2)))) * np.exp(-np.pi*np.power(QZ-QQ, 2)/np.power(sigma, 2))

    # RST output computed with matrix multiplication

    z_tmp = PSI.conj().T @  p
    Z = z_tmp.reshape(I, W, order='F')

    return Z, m, q, z

