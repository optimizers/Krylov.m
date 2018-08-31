function [x, stats] = cgls_spot(A, b, varargin)
%% CGLS
% Solve the regularized linear least-squares problem
%
%       minimize ||b - Ax||² + lambda * ||x||²
%
% using the Conjugate Gradient (CG) method, where lambda >= 0 is a
% regularization parameter. 
%
% CGLS is formally equivalent to applying the conjugate gradient method to
% the normal equations
%
%       (A'A + lambda*I) x = A'b
%
% but should be more stable.
%
% CGLS produces monotonic residuals ||r|| but not optimality residuals ||A'r||. 
% It is also formally equivalent to LSQR though LSQR should be expected 
% to be more stable on ill-conditioned or poorly scaled problems.
%
% This implementation is the standard formulation, as recommended by
% A. Björck, T. Elfving and Z. Strakos, Stability of Conjugate Gradient and
% Lanczos Methods for Linear Least Squares Problems.
%
% Translated from The Julia implementation 
% of Dominique Orban <dominique.orban@gerad.ca>
% Montréal, QC, June 2018
% -------------------------------------------------------------------------
% 
%       [x, stats] = cgls(A, b, 'name', value, ...)
%
% Input
%     A - Double matrix or spot operator containing the system matrix
%     b - Column vector
% Options
%     M       - Double matrix or spot operator representing the 
%             preconditioning operator (default: identity)
%     lambda  - Non negative regularization parameter (default: 0)
%     atol    - Absolute stopping tolerance for the optimality 
%             residual norm ||Ar|| (default: 1e-8)
%     rtol    - Relative stopping tolerance for the optimality 
%             residual norm ||Ar|| (default: 1e-6)
%     itmax   - Maximum number of iterations 
%             (default: number of rows of A + number of columns of A)
%     verbose - Print output if true (default: false)
% Output
%     x     - The last computed approximation of the solution
%     stats - Informations about the algorithm executions
%           solved:  true if optimality has been reached
%           rNorms:  history of the ||r|| residual norms
%           ArNorms: history of the ||Ar|| residual norms
%           status:  Situation at the end of the execution

%% Read input

[m, n] = size(A);
if size(b, 1) ~= m
    error('Inconsistent problem size');
end

p = inputParser;
p.PartialMatching = false;
p.addParameter('M'      , opEye(m));
p.addParameter('lambda' , 0);
p.addParameter('atol'   , 1e-8);
p.addParameter('rtol'   , 1e-6);
p.addParameter('itmax'  , n + m);
p.addParameter('verbose', false);

p.parse(varargin{:});

M = p.Results.M;
lambda = p.Results.lambda;
atol = p.Results.atol;
rtol = p.Results.rtol;
itmax = p.Results.itmax;
verbose = p.Results.verbose;

if verbose
    fprintf('CGLS: system of %d equations in %d variables\n', m, n);
end

%% Initialize variables
x = zeros(n, 1);
r = b;
bNorm = norm(b);

% Return if 0 is a solution
if bNorm == 0
    stats = struct('solved',  true, ...
                   'rNorms',  0.0,  ...
                   'ArNorms', 0.0,  ...
                   'status',  'x = 0 is a zero-residual solution');
    return
end

s = A' * (M * r);
p = s;
y = s' * s;
iter = 0;

rNorm = bNorm;
ArNorm = sqrt(y);
rNorms = [rNorm ; zeros(m + n, 1)];
ArNorms = [ArNorm ; zeros(m + n, 1)];
epsilon = atol + rtol * ArNorm;

if verbose
    fprintf('%5s %13s %13s\n', 'Aprod', '|A''r|', '|r|');
    fprintf('%5d %13.6e %13.6e\n', 1, ArNorm, rNorm);
end

solved = ArNorm <= epsilon;
tired = iter >= itmax;

%% Main loop
while ~(solved || tired)
    q = A * p;
    delta = q' * M * q;
    if lambda > 0
        delta = delta + lambda * (p' * p);
    end
    alpha = y / delta;
    
    x = x + alpha * p;
    r = r - alpha * q;
    s = A' * (M * r);
    if lambda > 0
        s = s - lambda * x;
    end
    
    yNext = s' * s;
    beta = yNext / y;
    p = s + beta * p;
    y = yNext;
    
    iter = iter + 1;
    rNorm = norm(r);
    ArNorm = sqrt(y);
    rNorms(iter + 1) = rNorm;
    ArNorms(iter + 1) = ArNorm;
    
    if verbose
        fprintf('%5d %13.6e %13.6e\n', 1 + 2 * iter, ArNorm, rNorm);
    end
    
    solved = ArNorm <= epsilon;
    tired = iter >= itmax;
end

%% Prepare output
if tired
    status = 'Maximum number of iterations exceeded';
else
    status = 'Solution good enough given atol and rtol';
end
stats = struct('solved',  solved,  ...
               'rNorms',  rNorms(1:iter + 1),  ...
               'ArNorms', ArNorms(1:iter + 1), ...
               'status',  status);
end


