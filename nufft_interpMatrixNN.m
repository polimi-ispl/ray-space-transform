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



function lambda = wrapTo2Pi(lambda)

positiveInput = (lambda > 0);
lambda = mod(lambda, 2*pi);
lambda((lambda == 0) & positiveInput) = 2*pi;
