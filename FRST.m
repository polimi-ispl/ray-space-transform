function [Z,m,q,z] = FRST(p,f,c,d,L,mbar,W,qbar,sigma,N,B)
%|
%| in
%|	p		[L,1] microphone signal in the frequency domain
%|	f		frequency in Hz
%|	c		speed of sound
%|	d		distance between adjacent microphones
%|	L       Number of microphones
%|	mbar    m axis sampling interval
%|  W       length of m axis
%|	qbar	q axis sampling interval
%|	sigma   gaussian window standar deviation
%|	N       oversampling factor (length of FFT)
%|	B       Number of nearest neighbors for the interpolation
%|          B = 1 (nearest neighbor interpolation), 
%|          B > 1 Keiser-Bessel minmax interpolation  see [1]
%|          
%| out
%|	Z       [W,I] FRST output 
%|	m       [W,1] m axis
%|	q       [I,1] q axis
%|  z       [L,1] microphone positions
%| 
%| [1] J. A. Fessler and B. P. Sutton,Nonuniform fast fourier transforms  using  min-max  interpolation
%|

%% Parameters

z = (0:d:d*(L-1))';                         % [L,1] microphone positions
m = ((0:mbar:(W-1)*mbar)-((W-1)/2*mbar))';  % [W,1] m axis
q = (0:qbar:z(end))';                       % [I,1] q axis
I = length(q);                              % number of frames

%% Signal Windowing

arg_psi = outer_sum(z,-q);
psi = exp(-pi*(arg_psi).^2 / sigma^2);      % Gaussian window with std sigma

%% FRST

% Set of nonuniformy spaced frequency locations
gamma = (2*pi*f/c)*((m)./(sqrt(1+m.^2)))*d; 
    
if B == 1
    % Nearest Neighbor interpolation matrix
    st = nufft_interpMatrixNN(gamma,L,N);
else
    % Keiser-Bessel minmax interpolation matrix see [1]
    st = nufft_interpMatrixKBminMax(gamma, L, B, N);
end

% Input signal scaling and windowing
pbar = repmat(d*st.sn*p,1,I).*psi;

% FRST output computed with the Nonuniform Fast Fourier Transform
Z = nufft(pbar,st).';

end

