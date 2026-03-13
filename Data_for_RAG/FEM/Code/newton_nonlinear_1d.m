function [u, info] = newton_nonlinear_1d(mesh, u0, pre, f_fun, k_fun, dk_fun, opts)
% newton_nonlinear_1d
%
% Solve the nonlinear 1D diffusion problem
%     R(u) = 0
% with
%     R_i(u) = \int k(u) u' N_i' dx - \int f N_i dx - (Neumann boundary contributions)
%
% using the Newton--Raphson method.
%
% Inputs:
%   mesh   : mesh struct with fields
%            - mesh.x, mesh.elem
%            - mesh.bound.dirichlet (node IDs)
%            - mesh.bound.gD        (scalar or vector values)
%            - mesh.bound.neumann   (optional)
%            - mesh.bound.gN        (optional)
%   u0     : initial guess (Ndof x 1)
%   pre    : struct with quadrature/basis info:
%            pre.N, pre.dN, pre.gp, pre.gw, pre.np, pre.ngp
%   f_fun  : handle for body load f(x)
%   k_fun  : handle for nonlinear conductivity k(u)
%   dk_fun : handle for dk/du
%   opts   : struct with fields
%            - maxIter : maximum Newton iterations
%            - tolRes  : residual tolerance (relative to initial)
%            - tolInc  : increment tolerance (relative)
%            - verbose : logical flag to print iteration info
%
% Outputs:
%   u      : converged solution (Ndof x 1)
%   info   : struct with Newton history
%            - info.resNorm    : residual norms per iteration
%            - info.incNorm    : increment norms per iteration
%            - info.nIter      : number of iterations performed
%            - info.converged  : true/false

Ndof = numel(mesh.x);
u    = u0(:);

% Make sure Dirichlet DOFs satisfy the prescribed values at the start
if isfield(mesh,'bound') && isfield(mesh.bound,'dirichlet')
    ids = mesh.bound.dirichlet(:);
    if isfield(mesh.bound,'gD')
        gD = mesh.bound.gD;
        if isscalar(gD)
            u(ids) = gD;
        else
            gD = gD(:);
            assert(numel(gD)==numel(ids), ...
                'Length of mesh.bound.gD must match number of Dirichlet nodes.');
            u(ids) = gD;
        end
    else
        u(ids) = 0;
    end
end

% Default options if not provided
if ~isfield(opts,'maxIter'), opts.maxIter = 20; end
if ~isfield(opts,'tolRes'),  opts.tolRes  = 1e-8; end
if ~isfield(opts,'tolInc'),  opts.tolInc  = 1e-10; end
if ~isfield(opts,'verbose'), opts.verbose = true; end

resNorm  = [];
incNorm  = [];
converged = false;

% First assembly to get initial residual norm
[R, KT] = assemble_nonlinear_1d(mesh, u, pre, f_fun, k_fun, dk_fun);
Rnorm0  = norm(R);
if Rnorm0 == 0
    Rnorm0 = 1;  % avoid divide-by-zero
end

if opts.verbose
    fprintf('  Iter %2d: ||R|| = %.3e (initial)\n', 0, Rnorm0);
end

for it = 1:opts.maxIter
    % Re-assemble residual and tangent at current iterate
    [R, KT] = assemble_nonlinear_1d(mesh, u, pre, f_fun, k_fun, dk_fun);
    Rnorm   = norm(R);
    resNorm(it,1) = Rnorm;
    
    if opts.verbose
        fprintf('  Iter %2d: ||R|| = %.3e (rel = %.3e)\n', ...
            it, Rnorm, Rnorm/Rnorm0);
    end
    
    % Check residual-based convergence
    if Rnorm / Rnorm0 < opts.tolRes
        converged = true;
        break;
    end
    
    % Build "increment" problem:
    %   KT * dU = -R
    % with Dirichlet increments dU = 0 at prescribed DOFs.
    mesh_inc = mesh;
    % For increments, prescribed value is zero at Dirichlet nodes
    mesh_inc.bound.gD = 0;
    
    % Use existing Dirichlet routine to eliminate increments at BCs
    [KTred, Fred, dU_D, free] = apply_dirichlet(KT, -R, mesh_inc);
    
    % Solve for free increments
    dU        = zeros(Ndof,1);
    dU(free)  = KTred \ Fred;
    % Dirichlet increments are enforced as zero via dU_D (mostly redundant)
    dU(~isnan(dU_D)) = dU_D(~isnan(dU_D));
    
    % Update
    u_new  = u + dU;
    dUnorm = norm(dU);
    incNorm(it,1) = dUnorm;
    
    % Increment-based convergence check (relative)
    relInc = dUnorm / (norm(u) + eps);
    if opts.verbose
        fprintf('           ||dU|| = %.3e, rel = %.3e\n', dUnorm, relInc);
    end
    
    u = u_new;
    
    if relInc < opts.tolInc
        converged = true;
        break;
    end
end

info.resNorm   = resNorm;
info.incNorm   = incNorm;
info.nIter     = it;
info.converged = converged;

if opts.verbose
    if converged
        fprintf('  Newton converged in %d iterations (relRes = %.3e).\n', ...
            it, resNorm(end)/Rnorm0);
    else
        fprintf('  Newton did NOT fully converge in %d iterations (relRes = %.3e).\n', ...
            it, resNorm(end)/Rnorm0);
    end
end
end