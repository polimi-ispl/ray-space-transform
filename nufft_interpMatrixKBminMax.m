function st = nufft_interpMatrixKBminMax(gamma, Nd, Jd, Kd)

%% Input Check

st.Nd = Nd;
st.Kd = Kd;
M = size(gamma,1);
st.M = M;

nufft_check_dft(gamma, Nd)

%% Find Keiser-Bessel alpha beta parameters

[alpha, beta] = nufft_alpha_kb_fit(Nd, Jd, Kd);

%% Find NUFFT scaling factor

Nmid = (Nd-1)/2;
tmp = nufft_scale(Nd, Kd, alpha, beta ,Nmid);
st.sn = sparse(1:Nd,1:Nd,tmp(:),Nd,Nd);

%% NUFFT interpolation matrix

tol = 0;
T = nufft_T(Nd, Jd, Kd, tol, alpha, beta); 
[r, arg] = nufft_r(gamma(:), Nd, Jd, Kd, alpha, beta); 
c = T * r;	clear T r
gam = 2*pi/Kd;
phase_scale = 1i * gam * (Nd-1)/2;

phase = exp(phase_scale * arg);
uu = phase .* c; 
uu = conj(uu);

% indices into oversampled FFT components
koff = floor(gamma(:) / gam - Jd/2);
kk = mod(outer_sum([1:Jd]', koff'), Kd) + 1;

mm = [1:M]; 
mm = mm(ones(prod(Jd),1),:); % [*Jd M]
% make sparse matrix
st.p = sparse(mm(:), double(kk(:)), double(uu(:)), M, prod(Kd));

function nufft_check_dft(gamma, Nd)
kk = gamma / (2*pi) .* repmat(Nd(:)', [size(gamma,1) 1]);
tol = 1e-6;
tmp = abs(round(kk) - kk);
if all(tmp(:) < tol) && any(gamma(:))
    warning('DFT samples has suboptimal accuracy')
end

function [alphas, beta] = nufft_alpha_kb_fit(N, J, K, varargin)

beta = 1;
arg.Nmid = (N-1)/2;

if N > 40
    arg.L = 13;		% empirically found to be reasonable
else
    arg.L = ceil(N/3);	% a kludge to avoid "rank deficient" complaints
end

nlist = [0:(N-1)]' - arg.Nmid;

% kaiser-bessel with previously numerically-optimized shape
sn_kaiser = 1 ./kaiser_bessel(nlist/K, J);

% use regression to match NUFFT with BEST kaiser scaling's
gam = 2*pi/K;
X = cos(beta * gam * nlist * [0:arg.L]); % [N L]
% coef = regress(sn_kaiser, X)';
coef = (X \ sn_kaiser)'; % this line sometimes generates precision warnings
if any(isnan(coef(:))) % if any NaN then big problem!
    coef = (pinv(X) * sn_kaiser)'; 
    if any(isnan(coef(:)))
        error 'bug: NaN coefficients';
    end
end
alphas = [real(coef(1)) coef(2:end)/2];

function sn = nufft_scale(N, K, alpha, beta, Nmid)

if ~isreal(alpha(1)), error 'need real alpha_0', end
L = length(alpha) - 1;

if L > 0
	sn = zeros(N,1);
	n = [0:(N-1)]';
	i_gam_n_n0 = 1i * (2*pi/K) * (n - Nmid) * beta;

	for l1=-L:L
		alf = alpha(abs(l1)+1);
		if l1 < 0, alf = conj(alf); end
		sn = sn + alf * exp(i_gam_n_n0 * l1);
	end

else
	sn = alpha * ones(N,1);
end



