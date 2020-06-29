function [Z,m,q,z] = RST(p,f,c,d,L,mbar,W,qbar,sigma)
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
%|          
%| out
%|	Z       [W,I] FRST output 
%|	m       [W,1] m axis
%|	q       [I,1] q axis
%|  z       [L,1] microphone positions
%| 

%% Parameters

z = (0:d:d*(L-1))';                         % [L,1] microphone positions
m = ((0:mbar:(W-1)*mbar)-((W-1)/2*mbar))';  % [W,1] m axis
q = (0:qbar:z(end))';                       % [I,1] q axis
I = length(q);                              % number of frames

%% RST

[M, Q] = meshgrid(m,q);
[MM,MZ] = meshgrid(M(:),z);
[QQ,QZ] = meshgrid(Q(:),z);

% RST transformation matrix
PSI = d*exp(1i*(2*pi*f/c)*MZ.*MM./sqrt(1+MM.^2)) .* exp(-pi*(QZ-QQ).^2/sigma^2);

% RST output computed with matrix multiplication
z_tmp = PSI'*p;
Z=reshape(z_tmp,I,W);

end

