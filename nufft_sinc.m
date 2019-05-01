 function y = nufft_sinc(x)
%function y = nufft_sinc(x)
%|
%| my version of "sinc" function, because matlab's sinc() is in a toolbox
%|

iz = find(x == 0); % indices of zero arguments
x(iz) = 1;
y = sin(pi*x) ./ (pi*x);
y(iz) = 1;