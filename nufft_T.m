 function T = nufft_T(N, J, K, tol, alpha, beta)
%function T = nufft_T(N, J, K, tol, alpha, beta, use_true_diric)
%|
%| Precompute the matrix T = [C' S S' C]\inv used in NUFFT.
%| This can be precomputed, being independent of frequency location.
%|
%| in
%|	N		# signal length
%|	J		# of neighbors
%|	K		# FFT length
%|	tol		tolerance for smallest eigenvalue
%|	alpha	[L+1]	Fourier coefficient vector for scaling
%|	beta		scale gamma=2*pi/K by this for Fourier series
%|
%| out
%|	T	[J J]	precomputed matrix
%|
%| Copyright 2000-1-9, Jeff Fessler, University of Michigan

if ~exist('tol','var') || isempty(tol)
	tol = 1e-7;
end
if ~exist('beta','var') || isempty(beta)
	beta = 1/2;
end

if N > K, fail 'N > K', end

% default with unity scaling factors
if ~exist('alpha','var') || isempty(alpha)

	% compute C'SS'C = C'C
	[j1,j2] = ndgrid(1:J, 1:J);
    cssc = nufft_sinc((j2 - j1) / (K/N));

% Fourier-series based scaling factors
else
	if ~isreal(alpha(1)), fail 'need real alpha_0', end
	L = length(alpha) - 1; % L
	cssc = zeros(J,J);
	[j1,j2] = ndgrid(1:J, 1:J);
	for l1 = -L:L
		for l2 = -L:L
			alf1 = alpha(abs(l1)+1);
			if l1 < 0, alf1 = conj(alf1); end
			alf2 = alpha(abs(l2)+1);
			if l2 < 0, alf2 = conj(alf2); end

			tmp = j2 - j1 + beta * (l1 - l2);
            tmp = nufft_sinc((tmp) / (K/N));
			cssc = cssc + alf1 * conj(alf2) * tmp;
		end
	end
end


% Inverse, or, pseudo-inverse

%smin = svds(cssc,1,0);
smin = min(svd(cssc));
if smin < tol % smallest singular value
	warning('Poor conditioning %g => pinverse', smin)
	T = pinv(cssc, tol/10);
else
	T = inv(cssc);
end
