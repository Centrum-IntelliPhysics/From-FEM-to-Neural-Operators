function [K,F] = assemble_system_1d(mesh, model, pre)
%ASSEMBLE_SYSTEM_1D  Assemble global K,F for 1D Poisson using degree-p FEM.
%
% Reference element & quadrature:
%   - Reference domain: xi ∈ [-1,1]
%   - Gauss points/weights: pre.gp (1×ngp), pre.gw (1×ngp)
%   - Basis on [-1,1]:      pre.N  (np×ngp), pre.dN (np×ngp)  [dN = dN/dxi]
%
% Geometry & mapping per element e:
%   nodes = mesh.elem(e,:)        (1×np, global ids)
%   coord = mesh.x(nodes)         (np×1, physical coordinates)
%   xa = coord(1), xb = coord(end), h = xb - xa, xc = 0.5*(xa+xb)
%   xq = xc + (h/2)*pre.gp        (1×ngp)
%   wJ = (h/2) * pre.gw           (1×ngp)
%   dNdx = (2/h) * pre.dN         (np×ngp)
%
% Inputs
%   mesh.dim  = 1
%   mesh.x    = (Ndof×1)
%   mesh.elem = (Ne×np)
%   model.k_fun(x), model.f_fun(x)  % function handles, accept vector x
%   pre.p, pre.np, pre.ngp, pre.gp, pre.gw, pre.N, pre.dN
%
% Outputs
%   K, F  : global stiffness and load
%
% Example
%   [gp,gw] = gauss_legendre(max(p+1,2));
%   [N,dN]  = basis1d_lagrange(p,gp);
%   pre = struct('p',p,'np',p+1,'ngp',numel(gp),'gp',gp,'gw',gw,'N',N,'dN',dN);
%   [K,F] = assemble_system_1d(mesh, model, pre);

x    = mesh.x;
elem = mesh.elem;
Ndof = size(x,1);
Ne   = size(elem,1);
np   = pre.np;

% Rough nnz guess: each element contributes up to np^2 entries
K = spalloc(Ndof, Ndof, Ne*np*np);
F = zeros(Ndof,1);

for e = 1:Ne
    nodes = elem(e,:);              % 1×np
    coord = x(nodes);               % np×1

    % Element geometry
    xa = coord(1); xb = coord(end);
    h  = xb - xa;
    xc = 0.5*(xa + xb);

    % Quadrature in physical space
    xq = xc + (h/2)*pre.gp;         % 1×ngp
    wJ = (h/2) * pre.gw;            % 1×ngp

    % Coefficients at Gauss points
    kq = model.k_fun(xq);           % 1×ngp
    fq = model.f_fun(xq);           % 1×ngp

    % Basis derivatives in physical coordinates
    dNdx = (2/h) * pre.dN;          % np×ngp

    % ---- Local matrices (vectorized over gauss points) ----
    % Ke = ∑_q (dNdx(:,q) * dNdx(:,q)') * (kq(q)*wJ(q))
    Wk   = (kq .* wJ);              % 1×ngp
    Ke   = dNdx * diag(Wk) * dNdx.';                 % np×np

    % Fe = ∑_q N(:,q) * (fq(q)*wJ(q))
    Fe   = pre.N * ( (fq .* wJ).' );                 % np×1

    % ---- Scatter-add to global ----
    K(nodes,nodes) = K(nodes,nodes) + Ke;
    F(nodes)       = F(nodes) + Fe;
end
end
