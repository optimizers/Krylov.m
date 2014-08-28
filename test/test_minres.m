% Example test program to solve
% [ M   A'] [x1]   [b1]
% [ A  -N ] [x2] = [b2]
% by applying MINRES with preconditioner diag(M,N).
% This test requires http://math.nist.gov/MatrixMarket/mmio/matlab/mmread.m
%
% 31 Jan 2014: First version: Dominique Orban <dominique.orban@gerad.ca>

problem = 'yao';
M = mmread(['../data/', problem, '/M.mtx']);
A = mmread(['../data/', problem, '/A.mtx']);
N = mmread(['../data/', problem, '/N.mtx']);
m = size(M,1);
n = size(N,1);

% Set up linear system Kx = b:
K = opHermitian([tril(M) , sparse(m,n) ; A , -tril(N)]);
e = ones(m+n, 1);
b = K * e;

% Set up preconditioner.
Minv = opChol(M); Ninv = opChol(N);
opts.M = [Minv , sparse(m,n) ; sparse(n,m) , Ninv];

% Solve with MINRES.
opts.show  = true;
opts.check = true;  % Check K=K' and opts.M is SPD.
opts.rtol  = 0;     % Force a stop on the energy norm of the direct error.
[x, flags, stats] = minres_spot(K, b, opts);
flags
stats
semilogy(stats.err_lbnds, 'b-'); hold on;   % Monotonically decreasing.
semilogy(stats.resvec, 'r-');               % Monotonically decreasing.
title('History');
legend('Error lower bound', 'Residual');

% Compute relative error in energy norm.
NE1 = N + A * Minv * A';        % Normal equations of the first kind.
NE2 = M + A' * Ninv * A;        % Normal equations of the second kind.
W = [NE2 , sparse(m,n) ; sparse(n,m) , NE1];
err = sqrt((x-e)' * W * (x-e));
fprintf('Absolute error in energy norm = %7.1e\n', err);
fprintf('Relative error in energy norm = %7.1e\n', err / sqrt(e' * W * e));
