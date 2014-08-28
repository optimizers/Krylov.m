function [x, flags, stats] = symmlq_spot(A, b, opts)

% [x, flags, stats] = symmlq_spot(A, b, opts)
%
% Spot version of symmlq developed by Dominique Orban.
% All optional input arguments go into the `opts` structure with the same name
% as in the original SYMMLQ. All original output arguments go into the `stats`
% structure with the same name as in the original SYMMLQ.
%
% The preconditioner is assumed to be symmetric and positive definite, i.e.,
% this method is equivalent to applying the standard SYMMLQ to the
% centrally-preconditioned system
%          L'AL y = L'b
% where LL' = inv(M) and Ly=x.
%
% A is a linear operator.
%
% opts.M is a linear operator representing the inverse of the preconditioner.
% More precisely, the product M*v should return the solution of the system
% Ky=v where K is the preconditioner. By default, opts.M is the identity.
%
% The original SYMMLQ documentation follows.
%
%        [ x, istop, itn, anorm, acond, rnorm, xnorm ] = ...
%          symmlq( n, b, aprodname, msolvename, iw, rw,  ...
%                  precon, shift, show, check, itnlim, rtol )
%
%  SYMMLQ is designed to solve the system of linear equations Ax = b
%  where A is an n by n symmetric matrix and b is a given vector.
%  A is accessed by means of a function call of the form
%               y = aprod ( n, x, iw, rw )
%  which must return the product y = Ax for any given vector x.
%  A positive-definite preconditioner M = C C' may optionally
%  be specified.  If precon is true, a function call of the form
%               y = msolve( n, x, iw, rw )
%  must solve the system My = x.
%  WARNING:   The files containing the functions 'aprod' and 'msolve'
%             must not be called aprodname.m or msolvename.m !!!!
%
%  For further information, type    help symdoc.

%  07 Jun 1989: Date of Fortran 77 version written by
%               Michael Saunders, Stanford University.
%  15 May 1990: MATLAB m-file symmlq.m derived from Fortran version
%               by Gerald N. Miranda Jr, UCSD.
%  02 Oct 1997: Move to CG point only if it is better than LQ point.
%               For interest, print qrnorm (= rnorm for minres).
%               Note that cgnorm is always one step ahead of qrnorm.
%  20 Oct 1999: Bug.  alfa1 = 0 caused Anorm = 0, divide by zero.
%               Need to estimate Anorm from column of Tk.
%  ------------------------------------------------------------------
%

%  Retrieve input arguments.
shift = 0;
show = true;
check = false;
n = size(A,1);
itnlim = 2*n;
rtol = 1.0e-12;
M = opEye(n);
store_resids = true;
resids = [];
store_iterates = false;
iterates = [];
if nargin > 2
  if isfield(opts, 'shift')
    shift = opts.shift;
  end
  if isfield(opts, 'show')
    show = opts.show;
  end
  if isfield(opts, 'print')
    show = opts.print;
  end
  if isfield(opts, 'check')
    check = opts.check;
  end
  if isfield(opts, 'itmax')  % For consistency with other kernels.
    itmax = opts.itmax;
  end
  if isfield(opts, 'itnlim')
    itnlim = opts.itnlim;
  end
  if isfield(opts, 'rtol')
    rtol = opts.rtol;
  end
  if isfield(opts, 'M')
    M = opts.M;
  end
  if isfield(opts, 'store_resids')
    store_resids = opts.store_resids;
  end
  if isfield(opts, 'store_iterates')
    store_iterates = opts.store_iterates;
  end
end

%  Initialize

first = 'Enter SYMMLQ.   ';
last  = 'Exit  SYMMLQ.   ';
space = ' ';
msg   =[' beta2 = 0.  If M = I, b and x are eigenvectors    '
        ' beta1 = 0.  The exact solution is  x = 0          '
        ' Requested accuracy achieved, as determined by rtol'
        ' Reasonable accuracy achieved, given eps           '
        ' x has converged to an eigenvector                 '
        ' acond has exceeded 0.1/eps                        '
        ' The iteration limit was reached                   '
        ' A does not define a symmetric matrix              '
        ' M does not define a symmetric matrix              '
        ' M does not define a pos-def preconditioner        '];

if show
   disp( space );
   disp( [ first 'Solution of symmetric Ax = b' ] );
   disp( sprintf( 'n      =  %3g     shift  =  %23.14e',...
                   n,                shift ) );
   disp( sprintf( 'itnlim =  %3g     eps    =  %11.2e    rtol   =  %11.2e\n' ,...
                   itnlim,           eps,                rtol ) );
end

istop  = 0;   ynorm  = 0;    w = zeros(n,1);  acond = 0;
itn    = 0;   xnorm  = 0;    x = zeros(n,1);  done  = false;
anorm  = 0;   rnorm  = 0;    v = zeros(n,1);

if store_iterates
  iterates = [iterates , x];
end

%  Set up y for the first Lanczos vector v1.
%  y is really beta1 * P * v1  where  P = C^(-1).
%  y and beta1 will be zero if b = 0.

y      = b;     r1     = b;
y      = M * r1;
b1     = y(1);  beta1  = r1' * y;

%  See if msolve is symmetric.

if check
   r2     = M * y;
   s      = y' * y;
   t      = r1' * r2;
   z      = abs( s - t );
   epsa   = (s + eps) * eps^(1/3);
   if z > epsa, istop = 7;  show = true;  done = true; end
end

%  Test for an indefinite preconditioner.
%  If b = 0 exactly, stop with x = 0.

if beta1 <  0, istop = 8;  show = true;  done = true;  end
if beta1 == 0,             show = true;  done = true;  end

%  Here and later, v is really P * (the Lanczos v).

if beta1 > 0
  beta1  = sqrt( beta1 );
  s      = 1 / beta1;
  v      = s * y;

  %  See if aprod is symmetric.

  y = A * v;
  if check
     r2   = M * y;
     s    = y' * y;
     t    = v' * r2;
     z    = abs( s - t );
     epsa = (s + eps) * eps^(1/3);
     if z > epsa, istop = 6;  done  = true;  show = true;  end
  end

  %  Set up y for the second Lanczos vector.
  %  Again, y is beta * P * v2  where  P = C^(-1).
  %  y and beta will be zero or very small if Abar = I or constant * I.

  y    = (- shift) * v + y;
  alfa = v' * y;
  y    = (- alfa / beta1) * r1 + y;

  %  Make sure  r2  will be orthogonal to the first  v.

  z  = v' * y;
  s  = v' * v;
  y  = (- z / s) * v + y;
  r2 = y;

  y = M * r2;
  oldb   = beta1;
  beta   = r2' * y;
  if beta < 0, istop = 8; show = true;  done = true; end

  %  Cause termination (later) if beta is essentially zero.

  beta  = sqrt( beta );
  if beta <= eps, istop = -1; end

  %  See if the local reorthogonalization achieved anything.

  denom = sqrt( s ) * norm( r2 )  +  eps;
  s     = z / denom;
  t     = v' * r2;
  t     = t / denom;

  if show
     disp( space );
     disp( sprintf( 'beta1 =  %10.2e   alpha1 =  %9.2e', beta1, alfa ) );
     disp( sprintf( '(v1, v2) before and after  %14.2e', s ) );
     disp( sprintf( 'local reorthogonalization  %14.2e\n', t ) );
  end

  %  Initialize other quantities.

  cgnorm = beta1;     rhs2   = 0;       tnorm  = alfa^2 + beta^2;
  gbar   = alfa;      bstep  = 0;       ynorm2 = 0;
  dbar   = beta;      snprod = 1;       gmax   = abs( alfa ) + eps;
  rhs1   = beta1;     x1cg   = 0;       gmin   = gmax;
  qrnorm = beta1;

end  % of beta1 > 0

resids = [resids ; beta1];

if show
  head1 = '   Itn     x(1)(cg)  normr(cg)  r(minres)';
  head2 = '    bstep    anorm    acond';
  disp( [head1 head2] )

  str1 = sprintf ( '%6g %12.5e %10.3e', itn, x1cg, cgnorm );
  str2 = sprintf ( ' %10.3e  %8.1e',    qrnorm, bstep/beta1 );
  disp( [str1 str2] )
end

%  ------------------------------------------------------------------
%  Main iteration loop.
%  ------------------------------------------------------------------
%  Estimate various norms and test for convergence.

if ~done
   while itn < itnlim
      itn    = itn  +  1;
      anorm  = sqrt( tnorm  );
      ynorm  = sqrt( ynorm2 );
      epsa   = anorm * eps;
      epsx   = anorm * ynorm * eps;
      epsr   = anorm * ynorm * rtol;
      diag   = gbar;

      if diag == 0, diag = epsa; end

      lqnorm = norm( [rhs1; rhs2] );
      qrnorm = snprod * beta1;
      cgnorm = qrnorm * beta / abs( diag );

      resids = [resids ; lqnorm];

%     Estimate  Cond(A).
%     In this version we look at the diagonals of  L  in the
%     factorization of the tridiagonal matrix,  T = L*Q.
%     Sometimes, T(k) can be misleadingly ill-conditioned when
%     T(k+1) is not, so we must be careful not to overestimate acond.

      if lqnorm < cgnorm
         acond  = gmax / gmin;
      else
         denom  = min( gmin, abs( diag ) );
         acond  = gmax / denom;
      end

      zbar   = rhs1 / diag;
      z      = (snprod * zbar + bstep) / beta1;
      x1lq   = x(1) + b1 * bstep / beta1;
      x1cg   = x(1) + w(1) * zbar  +  b1 * z;

%     See if any of the stopping criteria are satisfied.
%     In rare cases, istop is already -1 from above (Abar = const * I).

      if istop == 0
         if itn    >= itnlim , istop = 5; end
         if acond  >= 0.1/eps, istop = 4; end
         if epsx   >= beta1  , istop = 3; end
         if cgnorm <= epsx   , istop = 2; end
         if cgnorm <= epsr   , istop = 1; end
      end
%     ==================================================================
%     See if it is time to print something.

      prnt = 0;
      if n      <= 40         ,   prnt = 1; end
      if itn    <= 10         ,   prnt = 1; end
      if itn    >= itnlim - 10,   prnt = 1; end
      if rem(itn,10) == 0     ,   prnt = 1; end
      if cgnorm <= 10.0*epsx  ,   prnt = 1; end
      if cgnorm <= 10.0*epsr  ,   prnt = 1; end
      if acond  >= 0.01/eps   ,   prnt = 1; end
      if istop  ~= 0          ,   prnt = 1; end
%     if ~show                ,   prnt = 0; end

      if show & prnt == 1
         str1 = sprintf ( '%6g %12.5e %10.3e', itn, x1cg, cgnorm );
         str2 = sprintf ( ' %10.3e  %8.1e',    qrnorm, bstep/beta1 );
         str3 = sprintf ( ' %8.1e %8.1e',      anorm, acond );
         disp( [str1 str2 str3] )
      end
      if istop > 0, break; end

%     Obtain the current Lanczos vector  v = (1 / beta)*y
%     and set up  y  for the next iteration.

      if istop ~= 0, break; end

      s      = 1/beta;
      v      = s * y;
      y      = A * v;
      y      = (- shift) * v + y;
      y      = (- beta / oldb) * r1 + y;
      alfa   = v' * y;
      y      = (- alfa / beta) * r2 + y;
      r1     = r2;
      r2     = y;
      y      = M * r2;
      oldb   = beta;
      beta   = r2' * y;

      if beta < 0, istop = 6;  break;  end
      beta   = sqrt( beta );
      tnorm  = tnorm  +  alfa^2  +  oldb^2  +  beta^2;

%     Compute the next plane rotation for Q.

      gamma  = sqrt( gbar^2 + oldb^2 );
      cs     = gbar / gamma;
      sn     = oldb / gamma;
      delta  = cs * dbar  +  sn * alfa;
      gbar   = sn * dbar  -  cs * alfa;
      epsln  = sn * beta;
      dbar   =            -  cs * beta;

%     Update  X.

      z      = rhs1 / gamma;
      s      = z*cs;
      t      = z*sn;
      x      = s*w  +  t*v  +  x;
      w      = sn*w - cs*v;

      if store_iterates
        iterates = [iterates , x];
      end

%     Accumulate the step along the direction  b, and go round again.

      bstep  = snprod * cs * z  +  bstep;
      snprod = snprod * sn;
      gmax   = max( gmax, gamma );
      gmin   = min( gmin, gamma );
      ynorm2 = z^2  +  ynorm2;
      rhs1   = rhs2  -  delta * z;
      rhs2   =       -  epsln * z;
   end % while
%  ------------------------------------------------------------------
%  End of main iteration loop.
%  ------------------------------------------------------------------

%  Move to the CG point if it seems better.
%  In this version of SYMMLQ, the convergence tests involve
%  only cgnorm, so we're unlikely to stop at an LQ point,
%  EXCEPT if the iteration limit interferes.

   if cgnorm < lqnorm
      zbar   = rhs1 / diag;
      bstep  = snprod * zbar  +  bstep;
      ynorm  = sqrt( ynorm2  +  zbar^2 );
      x      = zbar * w + x;
   end

%  Add the step along  b.

   bstep  = bstep / beta1;
   y      = b;
   y      = M * b;
   x      = bstep * y + x;

   if store_iterates
    iterates = [iterates , x];
  end

%  Compute the final residual,  r1 = b - (A - shift*I)*x.

   y      = A * x;
   y      = (- shift) * x + y;
   r1     = b - y;
   rnorm  = norm ( r1 );
   xnorm  = norm (  x );

   resids = [resids ; rnorm];
end % if

%  ==================================================================
%  Display final status.
%  ==================================================================
if show
   disp( space )
   disp( [ last sprintf( ' istop   =  %3g               itn   =  %5g',...
                           istop,                      itn  ) ] )
   disp( [ last sprintf( ' anorm   =  %12.4e      acond =  %12.4e' ,...
                           anorm,                 acond  ) ] )
   disp( [ last sprintf( ' rnorm   =  %12.4e      xnorm =  %12.4e' ,...
                           rnorm,                 xnorm  ) ] )
   disp( [ last msg(istop+2,:) ])
end

flags.solved = istop;
flags.niters = itn;

stats.istop = istop;
stats.msg   = msg(istop+2,:);
stats.anorm = anorm;
stats.acond = acond;
stats.rnorm = rnorm;
stats.xnorm = xnorm;
stats.resids = resids;
stats.iterates = iterates;

% end SYMMLQ
