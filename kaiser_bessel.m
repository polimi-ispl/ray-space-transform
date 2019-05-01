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


