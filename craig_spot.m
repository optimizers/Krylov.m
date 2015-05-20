function [x, y, flags, stats] = craig_spot(A, b, opts)

%        [x, y, flags, stats] = craig_spot(A, b, opts);
%
% Spot version of CRAIG developed by Dominique Orban.
% All optional input arguments go into the `opts` structure with the same name
% as in the original CRAIG. All original output arguments go into the `stats`
% structure with the same name as in the original CRAIG.
%
% Preconditioners M and N may be provided via `opts.M` and `opts.N` and are assumed
% to be symmetric and positive definite. If `opts.sqd` is set to `true`, we solve
% the symmetric and quasi-definite system
% [ E   A' ] [ x ]   [ 0 ]
% [ A  -F  ] [ y ] = [ b ],
% where E = inv(M) and F = inv(N).
%
% If `opts.sqd` is set to `false` (the default), we solve the symmetric and
% indefinite system
% [ E   A' ] [ x ]   [ 0 ]
% [ A   0  ] [ y ] = [ b ].
% In this case, `opts.N` can still be specified and inv(N) indicates the norm
% in which `y` should be measured.
%
% A is a linear operator.
%
% opts.M is a linear operator representing the inverse of E.
% More precisely, the product M*v should return the solution of the system
% Ey=v. By default, opts.M is the identity.
%
% 28 Aug 2014: Spot version created by Dominique Orban <dominique.orban@gerad.ca>
% Spot may be obtained from https://github.com/mpf/spot
%-----------------------------------------------------------------------
%        [ x, y, istop, itn, rnorm, Anorm, Acond, xnorm ]...
%        = craigSOL( m, n, A, b, atol, btol, conlim, itnlim, show )
%
% CRAIG finds a solution x to the linear equation Ax = b, where
% A is a real matrix with m rows and n columns, and b is a real
% m-vector.  If A is square and nonsingular, CRAIG finds the
% unique solution x = A(inv)b.
% If the system Ax = b is under-determined (i.e. there are many
% solutions), CRAIG finds the solution of minimum Euclidean length,
% namely    x = A inv(A A') b.  Thus, CRAIG solves the problem
%
%          min  x'x  subject to  Ax = b.
%
% y returns a vector satisfying A'y = x.  Hence AA'y = b.
%
% A is an m by n matrix (ideally sparse),
% or a function handle such that
%    y = A(x,1) returns y = A*x   (where x will be an n-vector);
%    y = A(x,2) returns y = A'*x  (where x will be an m-vector).

%-----------------------------------------------------------------------
% CRAIG uses an iterative (conjugate-gradient-like) method.
% For further information, see
% 1. C. C. Paige and M. A. Saunders (1982a).
%    LSQR: An algorithm for sparse linear equations and sparse least squares,
%    ACM TOMS 8(1), 43-71.

% 08 Apr 2003: First craig.m derived from Fortran 77 version of craig1.for.
%              Michael Saunders, Systems Optimization Laboratory,
%              Dept of MS&E, Stanford University.
% 09 Apr 2003: Experimenting on singular systems (for inverse iteration).
%              Separately, on full-rank Ax = b, "Acond" seems to
%              over-estimate cond(A) drastically.
% 02 Oct 2006: Output y such that x = A'y (already in the f77 version).
% 15 Aug 2014: A can now be a matrix or a function handle, as in lsqrSOL.
%-----------------------------------------------------------------------

[m, n] = size(A);

% Retrieve input arguments.
damp = 0;
atol = 1.0e-6;
btol = 1.0e-6;
etol = 1.0e-6;
conlim = 1.0e+8;
itnlim = 2*max(m,n);
show = false;
% resvec = [];
% Aresvec = [];

M = opEye(m);
M_given = false;
N = opEye(n);
N_given = false;
window = 5;
err_vector = zeros(window,1);    % Lower bounds on direct error in energy norm.
err_lbnds = [];                  % History of values of err_lbnds.
err_lbnd_small = false;

if nargin > 2
  if isfield(opts, 'damp')
    damp = opts.damp;
  end
  if isfield(opts, 'atol')
    atol = opts.atol;
  end
  if isfield(opts, 'btol')
    btol = opts.btol;
  end
  if isfield(opts, 'etol')
    etol = opts.etol;
  end
  if isfield(opts, 'conlim')
    conlim = opts.conlim;
  end
  if isfield(opts, 'itnlim')
    itnlim = opts.itnlim;
  end
  if isfield(opts, 'show')
    show = opts.show;
  end
  if isfield(opts, 'M')
    M = opts.M;
    M_given = true;
  end
  if isfield(opts, 'N')
    N = opts.N;
    N_given = true;
  end
  if isfield(opts, 'window')
    window = opts.window;
  end
  if isfield(opts, 'sqd')
    if opts.sqd & M_given & N_given
      damp = 1.0;
    end
  end
end

% Initialize.

msg=['The exact solution is  x = 0                              '
     'Ax - b is small enough, given atol, btol                  '
     'The system Ax = b seems to be incompatible                '
     'The estimate of cond(A) has exceeded conlim               '
     'Ax - b is small enough for this machine                   '
     'Cond(A) seems to be too large for this machine            '
     'The iteration limit has been reached                      '
     'The truncated direct error is small enough, given etol    '];

if show
   disp(' ')
   disp('CRAIG          minimum-length solution of  Ax = b')
   str1 = sprintf('The matrix A has %8g rows  and %8g cols', m, n);
   str3 = sprintf('atol = %8.2e                 conlim = %8.2e', atol, conlim);
   str4 = sprintf('btol = %8.2e                 itnlim = %8g'  , btol, itnlim);
   str5 = sprintf('etol = %8.2e                 window = %8d'  , etol, window);
   disp(str1);   disp(str3);   disp(str4);   disp(str5);
end

itn    = 0;
istop  = 0;
ctol   = 0;
if conlim > 0, ctol = 1/conlim; end
Anorm  = 0;
Acond  = 0;
dnorm  = 0;
xnorm  = 0;

%     Set beta(1) and u(1) for the bidiagonalization.
%         beta*u = b.

v      = zeros(n,1);
Nv     = zeros(n,1);
x      = zeros(n,1);
w1     = zeros(m,1);
if damp > 0
  w2 = zeros(n,1);
else
  % When damp > 0, it is cheaper to compute
  % y at the end of the main loop. We only
  % recur it when damp == 0.
  y = zeros(m,1);
end;

Mu     = b;
u      = M * Mu;
beta   = sqrt(dot(u, Mu));
bnorm  = beta;
rnorm  = beta;
if beta==0, disp(msg(1,:)); return, end
u      = (1/beta)*u;
Mu     = (1/beta)*Mu;

% More initialization.
% aanorm  is norm(L_k)**2, an estimate of norm(A)**2.  It is
%                alpha1**2  +  (alpha2**2 + beta2**2)  +  ...
% ddnorm  is norm(D_k)**2, an estimate of norm( (A'A)inverse ).
% xxnorm  is norm(x_k)**2  =  norm(z_k)**2.

delta  =  damp;
theta  =  beta;

aanorm =  0;
ddnorm =  0;
xxnorm =  0;
oldrho =  1;
z      = -1;

if show
   disp(' ')
   str1 = '   Itn      x(1)       rnorm      xnorm';
   str2 = '     Norm A   Cond A   alpha    beta';
   disp([str1 str2])
   str1   = sprintf( '%6g %12.5e',        itn, x(1)  );
   str2   = sprintf( ' %10.3e',  rnorm );
   disp([str1 str2])
end

%------------------------------------------------------------------
%     Main iteration loop.
%------------------------------------------------------------------
while itn < itnlim
  itn = itn + 1;

  % Perform the next step of the bidiagonalization to obtain the
  % next alpha, v, beta, u.  These satisfy the relations
  %      alpha*v  =  A'*u  -  beta*v.
  %       beta*u  =  A*v   -  alpha*u,

  Nv = A'*u   - beta*Nv;
  v  = N * Nv;
  alpha  = sqrt(dot(v, Nv));
  if alpha==0, istop = 2; disp(msg(istop+1,:)); break, end
  v      = (1/alpha)*v;
  Nv     = (1/alpha)*Nv;

% Form a rotation  Q(i,k+i)  such that
%               (alpha  delta) ( cs1  -sn1 )  =  (rho   0  )
%               (  v    fbar ) ( sn1   cs1 )     ( w   fhat)

  if damp > 0
    rho = norm([alpha, delta]);
    cs1 = alpha / rho;
    sn1 = delta / rho;
  else
    rho = alpha;
  end

  aanorm =   aanorm + alpha^2 + damp^2;
  z      = - (theta/rho)*z;

  if damp > 0
    t1   =   z * cs1;
    t2   =   z * sn1;
    x    =   x + t1 * v + t2 * w2;
    w2   =     - sn1* v + cs1* w2;
  else
    x    =   x + z*v;
  end

  t1     = - theta/oldrho;
  t2     =   z   /rho;
  t3     =   1   /rho;

  %w1     =   u + t1*w1;
  w1     = u - theta * w1;
  if damp == 0
    y = y + t2*w1;
  end;
  ddnorm =   ddnorm  + norm(t3*w1)^2;

  Mu = A*v    - alpha*Mu;
  u  = M * Mu;
  beta = sqrt(dot(u, Mu));
  if beta > 0
      u  = (1/beta)*u;
      Mu = (1/beta)*Mu;
  end

  if damp > 0
    theta  =  cs1 * beta;
    delbar = -sn1 * beta;
  else
    theta  =  beta;
  end

% Form a rotation  Q(k+i,k+i-1)  such that
%               (deltabar  damp) (  cs2  sn2 )  =  (0  delta)
%               (  fhat     0  ) ( -sn2  cs2 )     (f  fbar )

  if damp > 0
    delta = norm([delbar, damp]);
    cs2   = damp   / delta;
    sn2   = delbar / delta;
    w2    = sn2 * w2;
  end

  %===============================================================
  % Test for convergence.
  % We estimate various norms and then see if
  % the quantities test1 or test3 are suitably small.
  %===============================================================
  Anorm  =   sqrt( aanorm );
  Acond  =   sqrt( ddnorm )*Anorm;
  xxnorm =   xxnorm + z^2;
  xnorm  =   sqrt( xxnorm );
  if damp > 0
    rnorm =  abs( cs1 * beta * z );
  else
    rnorm =  abs( beta*z );
  end

  test1  =   rnorm/bnorm;
  test3  =   1    /Acond;
  t1     =   test1/(1 + Anorm*xnorm/bnorm);
  rtol   =   btol + atol*Anorm*xnorm/bnorm;

  % ∑ ζ² is:
  % - the square M-norm of x, and also
  % - the square (N + A inv(M) A')-norm of y.
  err_vector(mod(itn,window)+1) = z;
  if itn >= window
    err_lbnd = norm(err_vector);
    err_lbnds = [err_lbnds ; err_lbnd];
    err_lbnd_small = (err_lbnd <= etol * sqrt(xnorm));
  end

  % The following tests guard against extremely small values of
  % atol, btol  or  ctol.  (The user may have set any or all of
  % the parameters  atol, btol, conlim  to zero.)
  % The effect is equivalent to the normal tests using
  % atol = eps,  btol = eps,  conlim = 1/eps.

  if itn >= itnlim,  istop = 6; end
  if err_lbnd_small, istop = 7; end
  if test3 <= eps ,  istop = 5; end
  if t1    <= eps ,  istop = 4; end
  % Allow for tolerances set by the user.
  if test3 <= ctol,  istop = 3; end
  if test1 <= rtol,  istop = 1; end

  % See if it is time to print something.

  prnt = 0;
  if n     <= 40       , prnt = 1; end
  if itn   <= 10       , prnt = 1; end
  if itn   >= itnlim-10, prnt = 1; end
  if rem(itn,10) == 0  , prnt = 1; end
  if test3 <=  2*ctol  , prnt = 1; end
  if test1 <= 10*rtol  , prnt = 1; end
  if istop ~=  0       , prnt = 1; end

  if prnt == 1
      if show
          str1 = sprintf( '%6g %12.5e',       itn, x(1 ) );
          str2 = sprintf( ' %10.3e %10.3e', rnorm, xnorm );
          str4 = sprintf( ' %8.1e %8.1e',   Anorm, Acond );
          str5 = sprintf( ' %8.1e %8.1e',  alpha , beta  );
          disp([str1 str2 str4 str5])
      end
  end
  if istop > 0, break, end;
  oldrho = rho;
  aanorm = aanorm + beta^2;
end

% Transfer to the LSQR point and recover y.
if damp > 0
  z = -(theta / delta) * z;
  x = x + z * w2;
  y = opts.N * (b - A * x);
end

if show
   disp(' ')
   disp('CRAIG finished')
   disp(msg(istop+1,:))
   disp(' ')
   str1 = sprintf( 'istop =%8g    rnorm =%8.1e',   istop, rnorm );
   str2 = sprintf( 'Anorm =%8.1e',  Anorm );
   str3 = sprintf( 'itn   =%8g    xnorm =%8.1e',     itn, xnorm );
   str4 = sprintf( 'Acond =%8.1e',  Acond );
   disp([str1 '   ' str2])
   disp([str3 '   ' str4])
   disp(' ')
end

% Collect statistics.
stats.istop = istop;
stats.rnorm = rnorm;
stats.Anorm = Anorm;
stats.Acond = Acond;
stats.xnorm = xnorm;
stats.err_lbnds = err_lbnds;

flags.solved = (istop >= 0 & istop <= 1) | istop == 4;
flags.niters = itn;

end % function craigSOL

