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