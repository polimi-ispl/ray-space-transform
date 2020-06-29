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


function y = kaiser_bessel(u,J)

if ~exist('J','var'), J = 6; end

d = 1;
kb_m = 0;
alpha = 2.34 * J;

z = sqrt( (2*pi*(J/2)*u).^2 - alpha^2 );
nu = d/2 + kb_m;
y = (2*pi)^(d/2) .* (J/2)^d .* alpha^kb_m ./ besseli(kb_m, alpha) ...
    .* besselj(nu, z) ./ z.^nu;
y = real(y);
end

function X = nufft(x, st)
%function X = nufft(x, st)
%|
%| Compute d-dimensional NUFFT of signal/image x
%|
%| in
%|	x	[N1 N2 ... Nd (L)]	L input image(s) of size
%|						N1 x N2 x ... x Nd
%|	st	structure		precomputed by nufft_init()
%| out
%|	X	[M (L)]			output spectra
%|
%| Copyright 2003-5-30, Jeff Fessler, University of Michigan

%y = st.sn*x;
Xk = fft(x, st.Kd);
X = st.p * Xk;
end

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
end

function nufft_check_dft(gamma, Nd)
kk = gamma / (2*pi) .* repmat(Nd(:)', [size(gamma,1) 1]);
tol = 1e-6;
tmp = abs(round(kk) - kk);
if all(tmp(:) < tol) && any(gamma(:))
    warning('DFT samples has suboptimal accuracy')
end
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
end

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
end

function st = nufft_interpMatrixNN(om,Nd,Kd)

%% Input

st.Nd = Nd;
st.Kd = Kd;
M = size(om,1);
st.M = M;

%% NUFFT scaling factor
st.sn = sparse(1:Nd,1:Nd,ones(Nd,1),Nd,Nd);
%% NUFFT interpolation matrix

om = wrapTo2Pi(om(:));
DFT_w = 0:2*pi/Kd:2*pi/Kd*(Kd-1);
tmp = outer_sum(om,-DFT_w);
[~,idx_min] = min(abs(tmp),[],2);

uu = ones(M,1);
mm = [1:M];
st.p = sparse(mm(:),idx_min,uu,M,Kd);
end

function lambda = wrapTo2Pi(lambda)
positiveInput = (lambda > 0);
lambda = mod(lambda, 2*pi);
lambda((lambda == 0) & positiveInput) = 2*pi;
end

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
end

function y = nufft_sinc(x)
%function y = nufft_sinc(x)
%|
%| my version of "sinc" function, because matlab's sinc() is in a toolbox
%|

iz = find(x == 0); % indices of zero arguments
x(iz) = 1;
y = sin(pi*x) ./ (pi*x);
y(iz) = 1;
end

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
end

function ss = outer_sum(xx,yy)
%|function ss = outer_sum(xx,yy)
%|
%| compute an "outer sum" x + y'
%| that is analogous to the "outer product" x * y'
%|
%| in
%|	xx	[nx 1]
%|	yy	[1 ny]
%|		more generally: xx [(dim)] + yy [L,1] -> xx [(dim) LL]
%| out
%|	ss [nx ny]	ss(i,j) = xx(i) + yy(j)
%|
%| Copyright 2001, Jeff Fessler, University of Michigan

% for 1D vectors, allow rows or cols for backward compatibility
if ndims(xx) == 2 && min(size(xx)) == 1 && ndims(yy) == 2 && min(size(yy)) == 1
    nx = length(xx);
    ny = length(yy);
    xx = repmat(xx(:), [1 ny]);
    yy = repmat(yy(:)', [nx 1]);
    ss = xx + yy;
end
end