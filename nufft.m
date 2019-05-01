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

