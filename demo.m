%% Parameters

f = 1000;       % Frequency Hz
c = 340;        % Speed of sound m/s

% ULA parameters

d = 0.1;        % Distance between two adjacent microphones
L = 64;         % Number of microphones

% Ray space parameters for RST and FRST

mbar = 0.06;    % m axis sampling interval
W = 101;        % length of m axis
qbar = d;       % q axis sampling interval
sigma = 5*d;    % gaussian window standar deviation

% Additional parameters for FRST

N = 2*L;        % oversampling factor (length of FFT)

% Number of nearest neighbors for the interpolation
% B = 1 (nearest neighbor interpolation), 
% B > 1 Keiser-Bessel minmax interpolation  see
% J. A. Fessler and B. P. Sutton, Nonuniform fast fourier transforms  using  min-max  interpolation
B = 5;         

%% Microphone signals simulation

z = (0:d:d*(L-1))';
r = [1.5 z(round(L/2))]';                   % Source position
dist=pdist2([zeros(L,1) z], r');            % Distance between source and microphones
p = exp(-1i*2*pi*f/c*dist) ./ (4*pi*dist);  % Microphone signals (e.g. free-field Green's Function)

%% RST and FRST

% RST
Z = RST(p,f,c,d,L,mbar,W,qbar,sigma);

% FRST
[Z_hat,m,q] = FRST(p,f,c,d,L,mbar,W,qbar,sigma,N,B);

%% NMSE

nmse = NMSE(Z_hat,Z);
disp(['NMSE: ' num2str(nmse) ' dB'])

%% Plot


figure('DefaultAxesFontSize',18)
scatter(zeros(L,1), z,'filled'); hold on
scatter(r(1),r(2),'square','filled');
grid on
xlabel('$x\,$ [m]','interpreter','latex')
ylabel('$z\,$ [m]','interpreter','latex')
legend('ULA','Point source','interpreter','latex')
title('Setup','interpreter','latex')


figure('DefaultAxesFontSize',18)
colormap bone

subplot(2,2,1)
imagesc(m,q,abs(Z));
axis xy
xlabel('m','interpreter','latex')
ylabel('q','interpreter','latex')
title('$\vert$ RST $\vert$','interpreter','latex')

subplot(2,2,2)
imagesc(m,q,abs(Z_hat));
axis xy
xlabel('m','interpreter','latex')
ylabel('q','interpreter','latex')
title('$\vert$ FRST $\vert$','interpreter','latex')

subplot(2,2,3)
imagesc(m,q,angle(Z));
axis xy
xlabel('m','interpreter','latex')
ylabel('q','interpreter','latex')
title('$\angle$ RST','interpreter','latex')

subplot(2,2,4)
imagesc(m,q,angle(Z_hat));
axis xy
xlabel('m','interpreter','latex')
ylabel('q','interpreter','latex')
title('$\angle$ FRST','interpreter','latex')