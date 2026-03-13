function [R, KT] = assemble_nonlinear_1d(mesh, u, pre, f_fun, k_fun, dk_fun)
% assemble_nonlinear_1d
%
% Assemble global residual vector R(u) and global tangent matrix K_T(u)
% for the 1D nonlinear diffusion problem:
%     - (k(u) u'(x))' = f(x)
%
% Weak form:
%   Find u in V such that for all test functions v in V0,
%       ∫_a^b k(u) u'(x) v'(x) dx = ∫_a^b f(x) v(x) dx + (Neumann terms)
%
% Residual at node i:
%   R_i(u) = ∑_e R_i^{(e)}(u) - (Neumann contributions at i)
%
% Inputs:
%   mesh   : mesh struct with mesh.x (Ndof×1), mesh.elem (Ne×np)
%   u      : current global solution vector (Ndof×1)
%   pre    : quadrature / basis struct (N, dN, gp, gw, etc.)
%   f_fun  : handle for body load f(x)
%   k_fun  : handle for nonlinear diffusion k(u)
%   dk_fun : handle for dk/du
%
% Outputs:
%   R      : residual vector (Ndof×1), R(u) = 0 at solution
%   KT     : tangent (Jacobian) matrix (Ndof×Ndof)

x    = mesh.x;
elem = mesh.elem;
Ndof = numel(x);
Ne   = size(elem,1);

R  = zeros(Ndof,1);
KT = zeros(Ndof,Ndof);

for e = 1:Ne
    nodes = elem(e,:);       % local node indices
    xe    = x(nodes);        % element node coordinates (np×1)
    ue    = u(nodes);        % element nodal values (np×1)
    
    % Element residual and tangent
    [Re, KeT] = element_res_tan_nonlinear_1d(xe, ue, pre, f_fun, k_fun, dk_fun);
    
    % Assemble into global residual/tangent
    R(nodes)          = R(nodes)          + Re;
    KT(nodes,nodes)   = KT(nodes,nodes)   + KeT;
end

% Optional Neumann boundary conditions:
% We treat them as additional "external" contributions that must be subtracted
% from R (since our Re already includes body loads f, but not Neumann).
F_neu = zeros(Ndof,1);
F_neu = apply_neumann(F_neu, mesh);   % adds nodal Neumann loads if present
R     = R - F_neu;
end