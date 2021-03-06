% Example to test the CGLS on a simple problem
% We generate a few least squares problems and solve them with the
% conjugate gradient

ntest = 20; % Number of tests
logcond = 2; % singular values are between 1 and 10^logcond

n = 10;     % number of equations
m = 5;      % number of unknowns

fprintf('====================================\n');
fprintf('    TEST CGLS\n');
fprintf('====================================\n');

if m > n
    error('There should be at least as many equations as unknowns');
end

clear op;
op.atol = eps;
op.rtol = 1e-12;
op.itmax = 2 * (n + m);
op.verbose = false;

nsolved = 0;

for itest = 1:ntest
    % Create the matrix
    A = randn(n, m);
    b = randn(n, 1);
    
    [U, ~, V] = svd(A);
    R = [diag(10 .^ (logcond / 2 * rand(m, 1))) ; ...
        zeros(n - m, m)];
    A = U * R * V';
    
    % Set regularization parameter
    op.lambda = (itest - 1) / ntest * 1e-3;
    
    % Preconditioner
    D = eye(n, n);
    for i = 1:m
        D(i, i) = 1 / R(i, i);
    end
    op.M = U * D * U';
   
    % Run CGLS
    [x, info] = cgls_spot(A, b, op);
    
    % Check answer
    ArNorm0   = norm(A' * b);
    ArNormEnd = norm(A' * (op.M * (b - A * x)) - op.lambda * x);
    suffDec = ArNormEnd <= (op.atol + ArNorm0 * op.rtol);
    
    if info.solved
        if ~suffDec
            msg = 'not solved';
        else
            msg = 'solved';
            nsolved = nsolved + 1;
        end
    else % tired
        msg = 'tired';
    end
    fprintf('%3d  rcond = %9.2e  %10s\n', ...
        itest, rcond(A.'*A), msg);
end

fprintf('====================================\n');
fprintf('Tests: %4d     Solved: %4d\n', ntest, nsolved); 
