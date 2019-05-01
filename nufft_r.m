 function [rr, arg] = nufft_r(om, N, J, K, alpha, beta)
%function [rr, arg] = nufft_r(om, N, J, K, alpha, beta, use_true_diric)
%|
%| make NUFFT "r" vector
%|
%| in
%|	om	[M 1]	digital frequency omega in radians
%|	N		signal length
%|	J		# of neighbors used per frequency location
%|	K		FFT size (should be > N)
%|	alpha	[0:L]	Fourier series coefficients of scaling factors
%|	beta		scale gamma=2pi/K by this in Fourier series
%|				typically is K/N (me) or 0.5 (Liu)
%| out
%|	rr	[J M]	r vector for each frequency
%|	arg	[J M]	dirac argument for t=0
%|
%| Copyright 2001-12-13, Jeff Fessler, University of Michigan

if ~exist('alpha','var') || isempty(alpha)
	alpha = [1]; % default Fourier series coefficients of scaling factors
end
if ~exist('beta','var') || isempty(beta)
	beta = 0.5; % default is Liu version for now
end

M = length(om);
gam = 2*pi/K;
dk = om / gam - floor(om(:) / gam - J/2);
arg = outer_sum(-[1:J]', dk');			% [J M] diric arg for t=0

L = length(alpha) - 1;
if ~isreal(alpha(1)), error 'need real alpha_0', end

if L > 0
	rr = zeros(J,M);
	for l1 = -L:L
		alf = alpha(abs(l1)+1);
		if l1<0, alf = conj(alf); end
        r1 = nufft_sinc((arg + l1 * beta) / (K/N));
		rr = rr + alf * r1;			% [J M]
	end
else
    rr = nufft_sinc(arg / (K/N));
end
